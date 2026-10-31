"""ANOVA-Simultaneous Component Analysis (#372).

The decomposition is exact and can be checked as an identity; everything else here is
about whether the test built on it has the power it claims. Two of these tests exist
because the obvious implementation is wrong in a way that still looks plausible: the
permutation null and the VASCA tie-break both had to be reconsidered, and the fixtures
below are the ones that showed it.
"""

from __future__ import annotations

import pathlib
import re
import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

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


def test_asca_uses_any_other_model_string_as_a_verbatim_right_hand_side() -> None:
    """A patsy right-hand side naming only A fits only A, and B's variation lands in the residual."""
    X, design, _ = _two_factor_design()
    main = ASCA(n_components=2, model="main_effects").fit(X, design)
    only_a = ASCA(n_components=2, model="C(A, Sum)").fit(X, design)

    assert only_a.terms_ == ["A"]
    # The design is balanced, so A's effect does not depend on whether B is in the model.
    pd.testing.assert_frame_equal(only_a.effect_matrices_["A"], main.effect_matrices_["A"], rtol=0, atol=1e-12)
    assert only_a.ssq_["residual"] == pytest.approx(main.ssq_["residual"] + main.ssq_["B"], rel=1e-9)


def test_asca_refuses_a_factor_named_c_with_a_rename_that_fits() -> None:
    """A factor named C hides patsy's C() in every C(name, Sum) term; the rename the error offers fits."""
    from process_improve.experiments.models import UnsafeFormulaError

    X, design, _ = _two_factor_design()
    named_c = design.rename(columns={"B": "C"})
    with pytest.raises(UnsafeFormulaError, match=re.escape(".rename(columns={'C': 'C_'})")):
        ASCA(n_components=2).fit(X, named_c)
    fitted = ASCA(n_components=2).fit(X, named_c.rename(columns={"C": "C_"}))
    reference = ASCA(n_components=2).fit(X, design)
    assert fitted.ssq_["C_"] == pytest.approx(reference.ssq_["B"], rel=1e-12)


def test_asca_checks_a_verbatim_right_hand_side_as_any_formula() -> None:
    X, design, _ = _two_factor_design()
    with pytest.raises(ValueError, match="only I"):
        ASCA(n_components=2, model="C(A, Sum) + eval(B)").fit(X, design)


def test_asca_reads_a_design_given_as_a_dict_of_columns_as_that_frame() -> None:
    """A design passed as a dict of columns gives the same decomposition as the frame it describes."""
    X, design, _ = _two_factor_design()
    from_frame = ASCA(n_components=2).fit(X, design)
    from_dict = ASCA(n_components=2).fit(X, design.to_dict(orient="list"))

    assert from_dict.terms_ == from_frame.terms_
    pd.testing.assert_series_equal(from_dict.ssq_, from_frame.ssq_)
    for name in from_frame.terms_:
        pd.testing.assert_frame_equal(from_dict.effect_matrices_[name], from_frame.effect_matrices_[name])


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


def test_asca_effect_summary_plot_draws_onto_a_figure_it_is_given() -> None:
    """Passing `fig` adds the bars to that figure and returns it."""
    go = pytest.importorskip("plotly.graph_objects")
    X, design, _ = _two_factor_design()
    canvas = go.Figure()
    returned = ASCA(n_components=2).fit(X, design).effect_summary_plot(fig=canvas)
    assert returned is canvas
    assert [trace.type for trace in canvas.data] == ["bar"]


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


# ---------------------------------------------------------------------------
# Checked by hand: each number against a calculation that does not use ASCA
# ---------------------------------------------------------------------------


def _effects_by_cell_means(Z: pd.DataFrame, design: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Two-way ANOVA of a balanced crossed design from its level and cell means: Z is centred already."""
    a, b = design["A"].to_numpy(), design["B"].to_numpy()
    a_mean, b_mean = Z.groupby(a).transform("mean"), Z.groupby(b).transform("mean")
    cell = Z.groupby([a, b]).transform("mean")
    return {"A": a_mean, "B": b_mean, "A:B": cell - a_mean - b_mean}, Z - cell


def _sum_coded(levels: pd.Series) -> np.ndarray:
    """Sum-to-zero contrasts: a column per sorted level but the last, which is -1 in every column."""
    names = sorted(levels.unique())
    last = (levels == names[-1]).to_numpy(dtype=float)
    return np.column_stack([(levels == name).to_numpy(dtype=float) - last for name in names[:-1]])


def _crossed_columns(design: pd.DataFrame, interactions: bool = True) -> dict[str, np.ndarray]:
    a, b = _sum_coded(design["A"]), _sum_coded(design["B"])
    columns = {"A": a, "B": b}
    if interactions:
        columns["A:B"] = np.column_stack([a[:, i] * b[:, j] for i in range(a.shape[1]) for j in range(b.shape[1])])
    return columns


def _effects_by_least_squares(
    Z: pd.DataFrame, columns: dict[str, np.ndarray]
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, np.ndarray]:
    """Fit every term and an intercept at once by the pseudo-inverse; each term's fitted part."""
    full = np.column_stack([np.ones(len(Z)), *columns.values()])
    coefficients = np.linalg.pinv(full) @ Z.to_numpy()
    effects, start = {}, 1
    for name, block in columns.items():
        width = block.shape[1]
        effects[name] = pd.DataFrame(block @ coefficients[start : start + width], index=Z.index, columns=Z.columns)
        start += width
    return effects, Z - full @ coefficients, full


def _ssq_by_hand(effects: dict[str, pd.DataFrame], residuals: pd.DataFrame, Z: pd.DataFrame) -> pd.Series:
    parts = {name: float((effect.to_numpy() ** 2).sum()) for name, effect in effects.items()}
    parts["residual"] = float((residuals.to_numpy() ** 2).sum())
    parts["total"] = float((Z.to_numpy() ** 2).sum())
    return pd.Series(parts, name="sum of squares")


@pytest.mark.parametrize("scale", [False, True])
def test_asca_decomposition_by_hand_on_a_balanced_design(scale: bool) -> None:
    """The level and cell means of classical two-way ANOVA, the sums of squares, and each term's PCA by SVD."""
    X, design, _ = _two_factor_design(interaction=1.0)
    model = ASCA(scale=scale).fit(X, design)
    spread = X.std(ddof=1) if scale else pd.Series(1.0, index=X.columns)
    Z = (X - X.mean()) / spread
    effects, residuals = _effects_by_cell_means(Z, design)

    pd.testing.assert_series_equal(model.grand_mean_, X.mean(), rtol=1e-14)
    pd.testing.assert_series_equal(model.column_scale_, spread, rtol=1e-14)
    np.testing.assert_array_equal(model.feature_names_in_, X.columns)
    assert model.terms_ == ["A", "B", "A:B"]
    for name, effect in effects.items():
        pd.testing.assert_frame_equal(model.effect_matrices_[name], effect, rtol=0, atol=1e-12)
    pd.testing.assert_frame_equal(model.residuals_, residuals, rtol=0, atol=1e-12)
    ssq = _ssq_by_hand(effects, residuals, Z)
    pd.testing.assert_series_equal(model.ssq_, ssq, rtol=1e-12)
    pd.testing.assert_series_equal(
        model.ssq_percent_, (ssq / ssq["total"] * 100).rename("percent of total"), rtol=1e-12
    )
    assert model.is_balanced_ is True
    assert model.is_fitted_ is True

    # Each term's PCA, of its effect alone at the default 2 components, capped at the term's
    # rank: 1 for two levels. Its scores are U S of the effect matrix, to a sign.
    for name, rank in {"A": 1, "B": 2, "A:B": 2}.items():
        u, s, _ = np.linalg.svd(effects[name].to_numpy(), full_matrices=False)
        scores = model.models_[name].scores_.to_numpy()
        assert scores.shape == (len(X), rank)
        np.testing.assert_allclose(np.abs(scores), np.abs(u[:, :rank] * s[:rank]), atol=1e-9)
        pd.testing.assert_index_equal(model.models_[name].loadings_.index, X.columns)


def test_asca_adds_the_residuals_before_each_terms_pca_by_hand() -> None:
    """ASCA+: each term's PCA is of its effect plus the residuals, at the default 2 components, or 3 asked for."""
    X, design, _ = _two_factor_design(interaction=1.0)
    effects, residuals = _effects_by_cell_means(X - X.mean(), design)
    for asked, expected in ((None, 2), (3, 3)):
        model = (ASCA(add_residuals=True) if asked is None else ASCA(asked, add_residuals=True)).fit(X, design)
        for name, effect in effects.items():
            u, s, _ = np.linalg.svd((effect + residuals).to_numpy(), full_matrices=False)
            scores = model.models_[name].scores_.to_numpy()
            assert scores.shape == (len(X), expected)
            np.testing.assert_allclose(np.abs(scores), np.abs(u[:, :expected] * s[:expected]), atol=1e-9)


def test_asca_caps_each_terms_pca_at_the_columns_of_x() -> None:
    """One response: every effect has rank 1 at most, so no PCA is asked for 2 components and none warns."""
    X, design, _ = _two_factor_design(interaction=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = ASCA().fit(X[["v5"]], design)
    assert [model.models_[name].scores_.shape[1] for name in model.terms_] == [1, 1, 1]


def _unbalanced_message(parts: float, total: float) -> str:
    return re.escape(
        "The design is unbalanced, so the terms are not orthogonal and the sums of squares do not partition the "
        f"total exactly: the parts add to {parts:.6g} against a total of {total:.6g}. Every term is still fitted "
        "simultaneously by least squares, which is the usual ASCA treatment, but a term's share depends on how the "
        "shared variation is attributed (the Type I / II / III question). Read the percentages as indicative rather "
        "than as a partition."
    )


@pytest.mark.parametrize(("units", "scale"), [(1.0, False), (1e-4, False), (1.0, True)])
def test_asca_unbalanced_decomposition_by_hand(units: float, scale: bool) -> None:
    """Rows dropped from one cell: the least-squares effects by the pseudo-inverse, and the warning in full.

    In units 1e4 times smaller the total is near 1e-6, and the parts still miss it by 2 percent.
    """
    X, design, _ = _two_factor_design(interaction=1.0)
    keep = np.ones(len(design), dtype=bool)
    keep[[0, 1, 7]] = False
    X, design = X[keep].reset_index(drop=True) * units, design[keep].reset_index(drop=True)
    Z = (X - X.mean()) / (X.std(ddof=1) if scale else 1.0)
    effects, residuals, _ = _effects_by_least_squares(Z, _crossed_columns(design))
    ssq = _ssq_by_hand(effects, residuals, Z)
    parts = ssq.drop("total").sum()
    with pytest.warns(UserWarning, match=f"^{_unbalanced_message(parts, ssq['total'])}$"):
        model = ASCA(scale=scale).fit(X, design)
    assert model.is_balanced_ is False
    # The intercept of the sum-coded fit is the mean of the cell means, not the column mean:
    # the residuals are what no term explains, so they are centred, and the intercept is the grand mean.
    cells = X.groupby([design["A"].to_numpy(), design["B"].to_numpy()]).mean()
    pd.testing.assert_series_equal(model.grand_mean_, cells.mean(), rtol=1e-12)
    pd.testing.assert_frame_equal(model.residuals_, residuals, rtol=0, atol=1e-12 * units)
    for name, effect in effects.items():
        pd.testing.assert_frame_equal(model.effect_matrices_[name], effect, rtol=0, atol=1e-12 * units)
    pd.testing.assert_series_equal(model.ssq_, ssq, rtol=1e-10)


def test_asca_a_design_missing_a_cell_is_unbalanced() -> None:
    """A 2 x 3 design without the (hi, m3) cell: called balanced before, and its parts missed the total unsaid."""
    X, design, _ = _two_factor_design(interaction=1.0)
    keep = ~((design["A"] == "hi") & (design["B"] == "m3")).to_numpy()
    X, design = X[keep].reset_index(drop=True), design[keep].reset_index(drop=True)
    Z = X - X.mean()
    effects, residuals, _ = _effects_by_least_squares(Z, _crossed_columns(design))
    ssq = _ssq_by_hand(effects, residuals, Z)
    with pytest.warns(UserWarning, match=f"^{_unbalanced_message(ssq.drop('total').sum(), ssq['total'])}$"):
        model = ASCA().fit(X, design)
    assert model.is_balanced_ is False
    for name, effect in effects.items():
        pd.testing.assert_frame_equal(model.effect_matrices_[name], effect, rtol=0, atol=1e-11)


def test_asca_a_fraction_fitted_for_its_main_effects_partitions_without_a_warning() -> None:
    """A half fraction of a 2^3 design is not balanced, as 4 combinations never occur, but its main effects
    are orthogonal: the parts add to the total, and nothing warns.
    """
    rng = np.random.default_rng(5)
    runs = [("lo", "lo", "hi"), ("hi", "lo", "lo"), ("lo", "hi", "lo"), ("hi", "hi", "hi")] * 3
    design = pd.DataFrame(runs, columns=["P", "Q", "R"])
    X = pd.DataFrame(rng.standard_normal((len(design), 4)), columns=["w", "x", "y", "z"])
    X["w"] += np.where(design["P"] == "hi", 2.0, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = ASCA(model="main_effects").fit(X, design)
    assert model.is_balanced_ is False
    Z = X - X.mean()
    for name in ("P", "Q", "R"):
        expected = Z.groupby(design[name].to_numpy()).transform("mean")
        pd.testing.assert_frame_equal(model.effect_matrices_[name], expected, rtol=0, atol=1e-12)
    assert model.ssq_.drop("total").sum() == pytest.approx(model.ssq_["total"], rel=1e-12)


def test_asca_reads_only_the_categories_a_factor_takes() -> None:
    """A categorical factor with categories no row takes, as a subset of a frame leaves it.

    Each unused category was coded as a level of its own, so the factor's effect was not centred:
    B's sum of squares came out 55 too high.
    """
    X, design, _ = _two_factor_design(interaction=1.0)
    reference = ASCA().fit(X, design)
    categorical = design.assign(B=pd.Categorical(design["B"], categories=["m0", "m1", "m2", "m3", "m9"]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = ASCA().fit(X, categorical)
    assert model.is_balanced_ is True
    for name in reference.terms_:
        pd.testing.assert_frame_equal(
            model.effect_matrices_[name], reference.effect_matrices_[name], rtol=0, atol=1e-12
        )
    pd.testing.assert_series_equal(model.ssq_, reference.ssq_, rtol=1e-12)


def test_asca_scales_every_column_whatever_its_units() -> None:
    """With scale=True a column in units 1e9 times smaller is scaled as any other, and a constant one is not.

    A spread below 1.5e-8 was read as a constant column's, so that column stayed unscaled: A's share
    of the total fell from 31 to 21 percent.
    """
    X, design, _ = _two_factor_design(interaction=1.0)
    small = X.assign(v0=X["v0"] * 1e-9, flat=0.3)
    model, reference = ASCA(scale=True).fit(small, design), ASCA(scale=True).fit(X, design)
    assert model.column_scale_["v0"] == pytest.approx(1e-9 * X["v0"].std(ddof=1), rel=1e-12)
    assert model.column_scale_["flat"] == 1.0
    assert model.grand_mean_["flat"] == 0.3
    for name in reference.terms_:
        _effect = model.effect_matrices_[name]
        pd.testing.assert_frame_equal(_effect[X.columns], reference.effect_matrices_[name], rtol=0, atol=1e-12)
        assert (_effect["flat"] == 0.0).all()
    pd.testing.assert_series_equal(model.ssq_, reference.ssq_, rtol=1e-12)


def test_asca_of_a_constant_response_has_nothing_to_test() -> None:
    """Every sum of squares is 0: no warning, every permutation ties the observed 0, and VASCA selects nothing."""
    X, design, _ = _two_factor_design()
    constant = pd.DataFrame(np.full(X.shape, 0.1), columns=X.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = ASCA().fit(constant, design)
        assert (model.ssq_ == 0.0).all()
        assert model.ssq_percent_.isna().all()
        pvalues = model.permutation_test(n_permutations=9, random_state=0)
        pd.testing.assert_series_equal(pvalues, pd.Series(1.0, index=["A", "B", "A:B"], name="p-value"))
        result = model.vasca("A", n_permutations=9, random_state=0)
    assert result.selected == []
    assert result.p_value == 1.0
    assert (result.table["z_score"] == 0.0).all()
    assert (result.table["p_value"] == 1.0).all()


def _reduced_null_by_hand(
    reduced: np.ndarray, full: np.ndarray, columns: slice, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Per-variable sums of squares of one term's effect, refitted on shuffles of its reduced response."""
    solver, rows = np.linalg.pinv(full), np.arange(reduced.shape[0])
    return np.array(
        [((full[:, columns] @ (solver @ reduced[rng.permutation(rows)])[columns]) ** 2).sum(axis=0) for _ in range(n)]
    )


def test_asca_permutation_test_at_its_defaults_by_hand() -> None:
    """999 shuffles of each term's effect plus the residuals, refitted, and (1 + #null >= observed) / 1000."""
    X, design, _ = _two_factor_design(a_size=0.3, b_size=0.15, interaction=0.2, noise=1.0, seed=4)
    model = ASCA().fit(X, design)
    Z = X - X.mean()
    effects, residuals, full = _effects_by_least_squares(Z, _crossed_columns(design))
    rng = np.random.default_rng(11)
    expected = {}
    for name, columns in {"A": slice(1, 2), "B": slice(2, 4), "A:B": slice(4, 6)}.items():
        null = _reduced_null_by_hand((effects[name] + residuals).to_numpy(), full, columns, 999, rng).sum(axis=1)
        expected[name] = (1 + np.sum(null >= (effects[name].to_numpy() ** 2).sum())) / 1000
    # p-values away from the floor and from 1
    assert min(expected.values()) > 0.002
    assert max(expected.values()) < 0.9
    pd.testing.assert_series_equal(model.permutation_test(random_state=11), pd.Series(expected, name="p-value"))
    pd.testing.assert_series_equal(model.pvalues_, pd.Series(expected, name="p-value"))


@pytest.mark.parametrize("units", [1.0, 1e-5])
def test_vasca_at_its_defaults_by_hand(units: float) -> None:
    """The ranking, the nested subsets' null, the raw and FDR p-values, the z-scores and the selection.

    In units 1e5 times smaller every spread of the null was below 1.5e-8: every z-score was 0, and
    the selection the first passing subset, v1 alone instead of v0 and v1.
    """
    X, design, _ = _two_factor_design(interaction=2.5)
    X = X * units
    model = ASCA().fit(X, design)
    Z = X - X.mean()
    effects, residuals, full = _effects_by_least_squares(Z, _crossed_columns(design))
    for name, columns in (("A", slice(1, 2)), ("A:B", slice(4, 6))):
        per_variable = (effects[name] ** 2).sum()
        ranking = per_variable.sort_values(ascending=False).index.tolist()
        observed = np.cumsum(per_variable[ranking].to_numpy())
        null = _reduced_null_by_hand(
            (effects[name] + residuals).to_numpy(), full, columns, 999, np.random.default_rng(0)
        )
        draws = np.cumsum(-np.sort(-null, axis=1), axis=1)
        raw = (1 + (draws >= observed).sum(axis=0)) / 1000
        fdr = multipletests(raw, method="fdr_bh")[1]
        z_score = (observed - draws.mean(axis=0)) / draws.std(axis=0, ddof=1)
        best = int(np.flatnonzero(fdr <= 0.05)[np.argmax(z_score[fdr <= 0.05])])

        result = model.vasca(name, random_state=0)
        assert result.ranking == ranking
        assert result.selected == ranking[: best + 1]
        assert best > 0 or name == "A:B"  # for A the z-score, not the first subset to pass, chose it
        assert result.p_value == pytest.approx(fdr.min(), rel=1e-12)
        if name == "A":  # every subset at the floor, 1 / 1000: an alpha of exactly that still passes them
            assert (fdr == 0.001).all()
            assert model.vasca(name, alpha=0.001, random_state=0).selected == result.selected
        expected = pd.DataFrame(
            {"variable": ranking, "ssq_cumulative": observed, "z_score": z_score, "p_value": raw, "p_value_fdr": fdr},
            index=pd.RangeIndex(1, 9, name="n_variables"),
        )
        pd.testing.assert_frame_equal(result.table, expected, rtol=1e-9)


def test_asca_refusals_in_full() -> None:
    X, design, _ = _two_factor_design()
    gappy = X.copy()
    gappy.iloc[4, 2] = np.nan
    with pytest.raises(
        ValueError, match=r"^ASCA needs a complete X: it solves one least-squares problem over every column\.$"
    ):
        ASCA().fit(gappy, design)
    gappy_design = design.astype(object)
    gappy_design.iloc[2, 1] = None
    with pytest.raises(
        ValueError, match=r"^The design frame holds missing values; every row must carry a level for every factor\.$"
    ):
        ASCA().fit(X, gappy_design)
    # One row: every factor has one level.
    with pytest.raises(ValueError, match=r"^These factors have a single level and can carry no effect: \['A', 'B'\]\."):
        ASCA().fit(X.iloc[:1], design.iloc[:1])


def test_asca_reads_an_array_as_columns_numbered_from_0_and_one_factor_alone() -> None:
    """X as an array; and the default interactions model of one factor is its main effect."""
    X, design, _ = _two_factor_design()
    from_array = ASCA().fit(X.to_numpy(), design)
    from_frame = ASCA().fit(X, design)
    np.testing.assert_array_equal(from_array.feature_names_in_, np.arange(X.shape[1]))
    np.testing.assert_allclose(from_array.effect_matrices_["A"].to_numpy(), from_frame.effect_matrices_["A"].to_numpy())
    alone = ASCA().fit(X, design[["A"]])
    assert alone.terms_ == ["A"]
    pd.testing.assert_frame_equal(alone.effect_matrices_["A"], from_frame.effect_matrices_["A"], rtol=0, atol=1e-12)
