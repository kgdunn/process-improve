"""The model variant: excluded rows stay out of the fit and are projected through it; a group is explained."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from process_improve.multivariate import PCA, PLS, MCUVScaler
from process_improve.workbench import Recipe, SourceFile
from process_improve.workbench.variant import Variant

LDPE = Path(__file__).parents[2] / "src" / "process_improve" / "datasets" / "multivariate" / "LDPE" / "LDPE.csv"
X_COLUMNS = (
    "Tin",
    "Tmax1",
    "Tout1",
    "Tmax2",
    "Tout2",
    "Tcin1",
    "Tcin2",
    "z1",
    "z2",
    "Fi1",
    "Fi2",
    "Fs1",
    "Fs2",
    "Press",
)
Y_COLUMNS = ("Conv", "Mn", "Mw", "LCB", "SCB")
UPSET = (52, 53, 54)  # the last three batches of the LDPE data, the process upset


@pytest.fixture(scope="module")
def ldpe() -> pd.DataFrame:
    """Return the LDPE table, indexed by batch number."""
    return pd.read_csv(LDPE, index_col=0)


def synthetic(n: int = 40, k: int = 6, seed: int = 1) -> pd.DataFrame:
    """Return a two-component table with a little noise."""
    rng = np.random.default_rng(seed)
    t = rng.normal(size=(n, 2)) * [3.0, 1.5]
    x = t @ rng.normal(size=(2, k)) + 0.2 * rng.normal(size=(n, k))
    return pd.DataFrame(x, columns=[f"x{i}" for i in range(k)], index=pd.RangeIndex(n) + 1).assign(
        y=t[:, 0] - t[:, 1] + 0.1 * rng.normal(size=n)
    )


@pytest.mark.dataset
def test_excluded_rows_are_left_out_of_the_fit_and_projected_through_it(ldpe) -> None:
    fit = Variant("PCA", 3, x_columns=X_COLUMNS, excluded=UPSET).fit(ldpe)
    included = ldpe.index.difference(list(UPSET))
    scaler = MCUVScaler().fit(ldpe.loc[included, list(X_COLUMNS)])
    by_hand = PCA(n_components=3).fit(scaler.transform(ldpe.loc[included, list(X_COLUMNS)]))
    np.testing.assert_array_equal(fit.model.loadings_.to_numpy(), by_hand.loadings_.to_numpy())
    projected = by_hand.project(scaler.transform(ldpe.loc[list(UPSET), list(X_COLUMNS)]))
    rows = fit.rows()
    assert rows.index.equals(ldpe.index)
    assert rows["excluded"].loc[list(UPSET)].all()
    assert not rows["excluded"].loc[included].any()
    np.testing.assert_array_equal(rows.loc[list(UPSET), "SPE"].to_numpy(), projected.spe.to_numpy())
    np.testing.assert_array_equal(rows.loc[list(UPSET), "T2"].to_numpy(), projected.hotellings_t2.to_numpy())


@pytest.mark.dataset
def test_the_upset_batches_fall_outside_the_limits_of_a_model_that_never_saw_them(ldpe) -> None:
    """The reason to project rather than drop: the excluded batches can still be judged against the model."""
    fit = Variant("PCA", 3, x_columns=X_COLUMNS, excluded=UPSET).fit(ldpe)
    limits, last = fit.limits(0.95), fit.rows().loc[54]
    assert last["T2"] > limits["T2"]
    assert last["SPE"] > limits["SPE"]


def test_an_excluded_row_has_no_influence_on_the_model() -> None:
    table = synthetic()
    changed = table.copy()
    changed.loc[[3, 7]] *= 50.0
    first = Variant("PCA", 2, excluded=(3, 7), y_columns=("y",)).fit(table)
    second = Variant("PCA", 2, excluded=(3, 7), y_columns=("y",)).fit(changed)
    np.testing.assert_array_equal(first.model.loadings_.to_numpy(), second.model.loadings_.to_numpy())
    assert (second.rows().loc[[3, 7], "SPE"] > first.rows().loc[[3, 7], "SPE"]).all()


def test_pca_takes_every_numeric_column_but_y_when_none_is_named() -> None:
    table = synthetic().assign(label="a")
    x_block, y_block = Variant("PCA", 2, y_columns=("y",)).blocks(table)
    assert list(x_block.columns) == [f"x{i}" for i in range(6)]
    assert y_block is not None
    assert list(y_block.columns) == ["y"]


@pytest.mark.parametrize("kind", ["PCA", "PLS"])
def test_the_variant_s_code_fits_the_same_model_after_the_recipe_s_code(kind: str, tmp_path, monkeypatch) -> None:
    """The recipe's script, then the variant's code: the same model and the same projected rows, bit for bit."""
    content = synthetic().to_csv().encode()
    (tmp_path / "data.csv").write_bytes(content)
    recipe = Recipe(SourceFile.of("data.csv", content, index_col=0))
    variant = Variant(kind, 2, x_columns=tuple(f"x{i}" for i in range(6)), y_columns=("y",), excluded=(5, 9))
    fit = variant.fit(recipe.run(content))
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(recipe.to_python() + variant.to_python(), namespace)  # noqa: S102 - the test runs its own script
    model = namespace["model"]
    assert isinstance(model, PCA | PLS)
    np.testing.assert_array_equal(model.scores_.to_numpy(), fit.model.scores_.to_numpy())
    projected = namespace["projected"]
    np.testing.assert_array_equal(projected.spe.to_numpy(), fit.rows().loc[[5, 9], "SPE"].to_numpy())


def test_pls_rows_carry_the_projected_x_space_diagnostics() -> None:
    fit = Variant("PLS", 2, y_columns=("y",), excluded=(1, 2)).fit(synthetic())
    rows = fit.rows()
    assert list(rows.columns) == [*fit.model.scores_.columns, "T2", "SPE", "excluded"]
    assert rows.loc[[1, 2], "excluded"].all()
    assert np.isfinite(rows[["T2", "SPE"]].to_numpy()).all()
    assert 0 < sum(fit.r2x()) <= 1


def test_r2x_per_component_adds_up_to_the_cumulative_r2_of_the_fit() -> None:
    fit = Variant("PCA", 3, y_columns=("y",)).fit(synthetic())
    np.testing.assert_allclose(np.cumsum(fit.r2x()), fit.model.r2_cumulative_.to_numpy(), rtol=1e-10)


# ---------------------------------------------------------------------------
# How many components?
# ---------------------------------------------------------------------------


@pytest.mark.dataset
def test_cross_validation_is_the_library_s_selector_on_the_fitted_rows_only(ldpe) -> None:
    cv = Variant("PCA", 2, x_columns=X_COLUMNS, excluded=UPSET).cross_validate(ldpe)
    included = ldpe.index.difference(list(UPSET))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_hand = PCA.select_n_components(ldpe.loc[included, list(X_COLUMNS)], random_state=0)
    np.testing.assert_array_equal(cv.q2.to_numpy(), by_hand.q2.to_numpy())
    assert cv.n_components == by_hand.n_components
    assert (cv.held_out, cv.rule) == ("cells", "min")
    assert list(cv.r2.index) == list(cv.q2.index) == list(range(1, len(X_COLUMNS) + 1))
    assert cv.r2.is_monotonic_increasing


@pytest.mark.slow
def test_pls_cross_validation_weighs_the_targets_alike_as_the_model_does() -> None:
    """A target in large units would otherwise decide the count alone; the model gives each unit variance."""
    table = synthetic().assign(big=lambda d: 1000 * (d["x0"] - d["x1"]) + 50 * np.sin(np.arange(len(d))))
    cv = Variant("PLS", 1, y_columns=("y", "big")).cross_validate(table)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = PLS.select_n_components(table.drop(columns=["y", "big"]), table[["y", "big"]], random_state=0)
    np.testing.assert_allclose(cv.q2.to_numpy(), raw.r2y_validated["scaled_total"].to_numpy(), rtol=1e-8)
    assert (cv.held_out, cv.folds) == ("rows", raw.per_fold_rmsecv.shape[1])


def test_the_number_of_components_plays_no_part_in_cross_validation() -> None:
    one, three = (Variant("PCA", n, y_columns=("y",)).cross_validate(synthetic()) for n in (1, 3))
    np.testing.assert_array_equal(one.q2.to_numpy(), three.q2.to_numpy())
    assert one.n_components == three.n_components


@pytest.mark.parametrize("kind", ["PCA", pytest.param("PLS", marks=pytest.mark.slow)])
def test_the_cross_validation_s_code_repeats_it_after_the_variant_s_code(kind: str, tmp_path, monkeypatch) -> None:
    content = synthetic().to_csv().encode()
    (tmp_path / "data.csv").write_bytes(content)
    recipe = Recipe(SourceFile.of("data.csv", content, index_col=0))
    variant = Variant(kind, 2, y_columns=("y",), excluded=(5, 9))
    cv = variant.cross_validate(recipe.run(content))
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(recipe.to_python() + variant.to_python() + cv.to_python(), namespace)  # noqa: S102 - its own script
    np.testing.assert_array_equal(np.asarray(namespace["q2"]), cv.q2.to_numpy())
    assert namespace["cv"].n_components == cv.n_components  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Why is this group different?
# ---------------------------------------------------------------------------


@pytest.mark.dataset
def test_a_group_s_contributions_add_up_to_its_shift_in_average_score(ldpe) -> None:
    fit = Variant("PCA", 3, x_columns=X_COLUMNS, excluded=UPSET).fit(ldpe)
    group = [48, 49, 50, 51, *UPSET]
    explanation = fit.explain_group(group)
    scores = fit.rows()
    reference = [r for r in fit.model.scores_.index if r not in group]
    shift = scores.loc[group, [1, 2, 3]].mean() - scores.loc[reference, [1, 2, 3]].mean()
    np.testing.assert_allclose(explanation.score_difference().to_numpy(), shift.to_numpy(), rtol=1e-10)
    assert explanation.reference == reference
    assert explanation.top[0] == "z2", "the zone 2 inlet: what the LDPE upset is known for"


def test_the_explanation_puts_the_evidence_beside_each_bar() -> None:
    table = synthetic()
    table.loc[[4, 5, 6], "x2"] += 4 * table["x2"].std()
    fit = Variant("PCA", 2, y_columns=("y",)).fit(table)
    explanation = fit.explain_group([4, 5, 6], top=3)
    assert explanation.top[0] == "x2"
    assert list(explanation.contributions.index) == list(explanation.summary.index)
    assert list(explanation.evidence.columns) == [*explanation.top, "in_group", "in_reference"]
    assert explanation.evidence["in_group"].sum() == 3
    assert explanation.evidence.index.equals(table.index)
    row = explanation.summary.loc["x2"]
    assert row["difference"] == pytest.approx(row["group mean"] - row["reference mean"])
    assert row["difference in reference SDs"] > 2


def test_a_group_can_be_compared_with_a_reference_of_its_own() -> None:
    fit = Variant("PCA", 2, y_columns=("y",)).fit(synthetic())
    explanation = fit.explain_group([1, 2, 3], reference=[10, 11, 12], components=[2])
    assert list(explanation.contributions.columns) == ["PC2"]
    assert explanation.reference == [10, 11, 12]


def test_missing_cells_still_give_every_variable_a_contribution() -> None:
    table = synthetic()
    table.loc[[4, 12], "x1"] = np.nan
    fit = Variant("PCA", 2, y_columns=("y",), excluded=(12,)).fit(table)
    explanation = fit.explain_group([4, 5, 12])
    assert np.isfinite(explanation.contributions.to_numpy()).all()


@pytest.mark.parametrize(
    ("group", "reference", "message"),
    [
        ([999], None, "not in the table"),
        ([1], [999], "not in the table"),
        ([], None, "holds no rows"),
    ],
)
def test_an_explanation_of_rows_that_do_not_exist_says_so(group, reference, message) -> None:
    fit = Variant("PCA", 2, y_columns=("y",)).fit(synthetic())
    with pytest.raises(ValueError, match=message):
        fit.explain_group(group, reference=reference)


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "LDA"}, "Unknown model"),
        ({"n_components": 0}, "at least 1"),
        ({"kind": "PLS"}, "needs at least one Y"),
    ],
)
def test_a_variant_refuses_what_it_cannot_fit(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        Variant(**kwargs)


def test_a_variant_says_when_too_few_rows_remain() -> None:
    with pytest.raises(ValueError, match="too few for 2 components"):
        Variant("PCA", 2, y_columns=("y",), excluded=tuple(range(1, 39))).fit(synthetic())


def test_a_variant_refuses_text_columns_and_unknown_rows() -> None:
    table = synthetic().assign(label="a")
    with pytest.raises(ValueError, match="not numbers"):
        Variant("PCA", 2, x_columns=("x0", "label")).blocks(table)
    with pytest.raises(ValueError, match="not in the table"):
        Variant("PCA", 2, y_columns=("y",), excluded=(999,)).fit(synthetic())


def test_a_variant_round_trips_through_json() -> None:
    variant = Variant("PLS", 3, x_columns=("a", "b"), y_columns=("y",), excluded=(4, "B12"), name="Without B12")
    assert Variant.from_dict(json.loads(json.dumps(variant.to_dict()))) == variant


@pytest.mark.parametrize(
    "data",
    [
        {"kind": "PCA", "n_components": "2"},
        {"kind": "PCA", "n_components": True},
        {"kind": "PCA", "excluded": [None]},
        {"kind": "PCA", "x_columns": "Tin"},
        {"kind": ["PCA"]},
    ],
)
def test_a_variant_read_from_a_file_is_checked(data) -> None:
    with pytest.raises((TypeError, ValueError)):
        Variant.from_dict(data)


def test_a_variant_naming_a_missing_column_says_so() -> None:
    with pytest.raises(ValueError, match="no column absent"):
        Variant("PCA", 2, x_columns=("x0", "absent")).blocks(synthetic())


def test_the_code_of_a_variant_without_named_x_columns_takes_the_numeric_ones(tmp_path, monkeypatch) -> None:
    content = synthetic().assign(label="a").to_csv().encode()
    (tmp_path / "data.csv").write_bytes(content)
    recipe = Recipe(SourceFile.of("data.csv", content, index_col=0))
    variant = Variant("PCA", 2, y_columns=("y",))
    assert '.drop(columns=["y"]).select_dtypes("number")' in variant.to_python()
    assert "model.project(" not in variant.to_python(), "nothing excluded, nothing to project"
    fit = variant.fit(recipe.run(content))
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {}
    exec(recipe.to_python() + variant.to_python(), namespace)  # noqa: S102 - the test runs its own script
    np.testing.assert_array_equal(namespace["model"].loadings_.to_numpy(), fit.model.loadings_.to_numpy())


def test_a_group_of_every_fitted_row_has_nothing_to_compare_against() -> None:
    fit = Variant("PCA", 2, y_columns=("y",)).fit(synthetic())
    with pytest.raises(ValueError, match="nothing to compare the group against"):
        fit.explain_group(list(synthetic().index))
