"""Cross-validating a multi-block PLS by holding out rows.

The scheme is the same one :meth:`PLS.select_n_components` uses, and it is sound
here for the same reason: a held-out row's super score comes from its X-blocks
alone, so its Y never contributes to its own prediction. Having several X-blocks
does not touch that argument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from process_improve.multivariate.methods import MBPLS


def _two_block_data(seed: int = 0, n: int = 60, rank: int = 2) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Two X-blocks and a Y block driven by the same `rank` latent variables."""
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal((n, rank)) * np.array([5.0, 3.0, 1.5])[:rank]
    blocks = {
        "wide": pd.DataFrame(scores @ rng.standard_normal((rank, 8)) + rng.standard_normal((n, 8)) * 0.4),
        "narrow": pd.DataFrame(scores @ rng.standard_normal((rank, 3)) + rng.standard_normal((n, 3)) * 0.4),
    }
    y = pd.DataFrame(scores @ rng.standard_normal((rank, 2)) + rng.standard_normal((n, 2)) * 0.4)
    return blocks, y


def test_returns_the_documented_bunch() -> None:
    """Every field the docstring promises, with the shapes it promises."""
    blocks, y = _two_block_data()
    out = MBPLS.select_n_components(blocks, y, max_components=4, cv=5, n_repeats=2, random_state=0)

    assert list(out.rmsecv.index) == [1, 2, 3, 4]
    assert out.rmsecv.index.name == "n_components"
    assert out.se_rmsecv.shape == (4,)
    assert out.per_fold_rmsecv.shape == (4, 10)  # five folds, twice over
    assert list(out.r2y_validated.columns) == [*y.columns, "total"]
    assert out.cv_predictions.shape == y.shape
    assert out.selection_rule == "1se"
    assert 1 <= out.n_components <= 4


def test_holding_out_rows_costs_something() -> None:
    """The defining property: a held-out row is predicted worse than a fitted one.

    If the held-out Y were reaching its own prediction, the cross-validated value
    could match or beat the fit. It must not.

    The baseline is the fitted model's own predictions on the original Y scale,
    which is what the validated value is built from. It is deliberately not
    ``r2_y_cumulative_``: that is computed on the scaled Y, where every target
    carries equal weight, so the two are on different footings and differ by a
    wide margin on data whose targets have unequal spread.
    """
    blocks, y = _two_block_data()
    out = MBPLS.select_n_components(blocks, y, max_components=3, cv=5, n_repeats=2, random_state=0)
    values = y.to_numpy(dtype=float)
    tss = float(((values - values.mean(axis=0)) ** 2).sum())

    for a in (1, 2, 3):
        fitted = MBPLS(n_components=a).fit(blocks, y)
        predicted = np.asarray(fitted.diagnose(blocks).predictions, dtype=float)
        fitted_r2 = 1.0 - float(((values - predicted) ** 2).sum()) / tss
        validated = float(out.r2y_validated["total"].loc[a])
        assert validated < fitted_r2, f"{a} components: validated {validated} beat the fit {fitted_r2}"
        assert validated > 0, "the blocks do carry signal, so the model must beat the mean"


def test_finds_the_rank_the_blocks_carry() -> None:
    """Two latent variables generated the data, and the curve should say so."""
    blocks, y = _two_block_data(rank=2)
    out = MBPLS.select_n_components(blocks, y, max_components=5, cv=5, n_repeats=3, random_state=0)
    gains = np.diff([0.0, *out.r2y_validated["total"].tolist()])

    assert out.n_components <= 3
    assert gains[0] > 0.3, "the first component carries most of it"
    assert gains[2] < 0.02, "a third component adds nothing the blocks support"


def test_repeatable_and_seed_sensitive() -> None:
    """The same seed gives the same curve; a different one moves it."""
    blocks, y = _two_block_data()
    kwargs = {"max_components": 3, "cv": 5, "n_repeats": 2}
    first = MBPLS.select_n_components(blocks, y, random_state=0, **kwargs)
    again = MBPLS.select_n_components(blocks, y, random_state=0, **kwargs)
    other = MBPLS.select_n_components(blocks, y, random_state=7, **kwargs)

    assert np.allclose(first.rmsecv, again.rmsecv)
    assert not np.allclose(first.per_fold_rmsecv.to_numpy(), other.per_fold_rmsecv.to_numpy())


def test_accepts_a_splitter_and_ignores_n_repeats() -> None:
    """A pre-built splitter is used as given, as it is for PLS."""
    blocks, y = _two_block_data()
    out = MBPLS.select_n_components(blocks, y, max_components=2, cv=KFold(4), n_repeats=9, random_state=0)

    assert out.per_fold_rmsecv.shape == (2, 4), "four folds, not four times nine"


def test_component_count_capped_by_the_smallest_fold() -> None:
    """More components than a training fold can support are not evaluated."""
    blocks, y = _two_block_data(n=12)
    out = MBPLS.select_n_components(blocks, y, max_components=50, cv=4, n_repeats=1, random_state=0)

    smallest_train = 12 - 12 // 4
    assert len(out.rmsecv) <= smallest_train - 1


@pytest.mark.parametrize(
    ("blocks", "error", "message"),
    [
        ([], TypeError, r"dict of DataFrames.*got list"),
        ({}, ValueError, "at least one block"),
        ({"a": np.zeros((10, 3))}, TypeError, r"'a' must be a pandas DataFrame; got ndarray"),
    ],
)
def test_rejects_malformed_blocks(blocks: object, error: type[Exception], message: str) -> None:
    """A wrong type raises TypeError and a wrong value ValueError, each naming what arrived."""
    _, y = _two_block_data(n=10)
    with pytest.raises(error, match=message):
        MBPLS.select_n_components(blocks, y, max_components=1, cv=2)


def test_rejects_a_component_count_below_one() -> None:
    """Asking for no components is the caller's mistake, and the message names it.

    The cap is applied afterwards with ``max(1, ...)``, so a fold too small to support
    more components still supports one. Zero can only come from the argument.
    """
    blocks, y = _two_block_data(n=20)
    with pytest.raises(ValueError, match=r"max_components must be at least 1; got 0"):
        MBPLS.select_n_components(blocks, y, max_components=0, cv=2)


def test_rejects_a_block_of_the_wrong_length() -> None:
    """A block that does not line up with y is named, with both row counts."""
    blocks, y = _two_block_data(n=20)
    blocks["narrow"] = blocks["narrow"].iloc[:15]
    with pytest.raises(ValueError, match=r"'narrow' has 15 rows; y has 20"):
        MBPLS.select_n_components(blocks, y, max_components=1, cv=2)
