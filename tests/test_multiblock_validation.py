"""Argument and data guards of MBPCA and MBPLS, and the small paths beside them.

The two estimators share their block validation, constructor checks, projection
guards, SPE limits and score-plot guard, with identical messages, so those tests
run once per class. Everything fits a 10-row, two-block set in milliseconds.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from process_improve.multivariate._limits import spe_calculation
from process_improve.multivariate.methods import MBPCA, MBPLS

MultiblockModel = MBPCA | MBPLS


def _blocks(seed: int = 0, n: int = 10) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return two X-blocks, "a" (5 columns) and "b" (3 columns), and a 2-column Y, from 2 latent variables."""
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((n, 2))
    blocks = {
        "a": pd.DataFrame(
            latent @ rng.standard_normal((2, 5)) + 0.1 * rng.standard_normal((n, 5)),
            columns=[f"a{i}" for i in range(5)],
        ),
        "b": pd.DataFrame(
            latent @ rng.standard_normal((2, 3)) + 0.1 * rng.standard_normal((n, 3)),
            columns=[f"b{i}" for i in range(3)],
        ),
    }
    y = pd.DataFrame(latent @ rng.standard_normal((2, 2)) + 0.1 * rng.standard_normal((n, 2)), columns=["y1", "y2"])
    return blocks, y


def _fit(cls: type[MultiblockModel], blocks: dict, y: pd.DataFrame, **kwargs: object) -> MultiblockModel:
    """Fit either class: MBPCA takes no Y."""
    model = cls(**{"n_components": 2, **kwargs})
    return model.fit(blocks) if cls is MBPCA else model.fit(blocks, y)


@pytest.mark.parametrize("cls", [MBPCA, MBPLS])
@pytest.mark.parametrize(
    ("make_blocks", "error", "message"),
    [
        (lambda _: {}, TypeError, r"^X must be a non-empty dict\[str, pd\.DataFrame\]\.$"),
        (
            lambda blocks: {**blocks, "a": blocks["a"].to_numpy()},
            TypeError,
            r"^X\['a'\] must be a pandas DataFrame; got ndarray\.$",
        ),
        (
            lambda blocks: {**blocks, "b": blocks["b"].iloc[:9]},
            ValueError,
            r"^All X-blocks must have the same row count\. Block 'b' has 9 rows; expected 10\.$",
        ),
    ],
    ids=["no-blocks", "block-not-a-frame", "blocks-of-unequal-height"],
)
def test_fit_refuses_malformed_blocks(
    cls: type[MultiblockModel], make_blocks: Callable[[dict], dict], error: type[Exception], message: str
) -> None:
    """X must be a non-empty dict of DataFrames that all have the same number of rows."""
    blocks, y = _blocks()
    with pytest.raises(error, match=message):
        _fit(cls, make_blocks(blocks), y)


@pytest.mark.parametrize("cls", [MBPCA, MBPLS])
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"n_components": 0}, r"^n_components must be positive; got 0\.$"),
        ({"max_iter": 0}, r"^max_iter must be positive; got 0\.$"),
        (
            {"algorithm": "nipals", "missing_data_settings": {"md_tol": 10}},
            r"^Tolerance should not be too large\.$",
        ),
        (
            {"algorithm": "nipals", "missing_data_settings": {"md_tol": 1e-30}},
            r"^Tolerance must exceed machine precision\.$",
        ),
    ],
    ids=["no-components", "no-iterations", "md-tol-too-large", "md-tol-below-precision"],
)
def test_invalid_settings_are_refused(cls: type[MultiblockModel], arguments: dict, message: str) -> None:
    """Counts must be positive, and a NIPALS tolerance must sit between machine precision and 10."""
    blocks, y = _blocks()
    with pytest.raises(ValueError, match=message):
        _fit(cls, blocks, y, **arguments)


@pytest.mark.parametrize("cls", [MBPCA, MBPLS])
@pytest.mark.parametrize(
    ("action", "error", "message"),
    [
        (lambda model, blocks: model.diagnose(blocks["a"]), TypeError, r"^X must be a dict\[str, pd\.DataFrame\]\.$"),
        (
            lambda model, blocks: model.diagnose({"a": blocks["a"]}),
            ValueError,
            r"^Missing X-blocks for prediction: \['b'\]\.$",
        ),
        (
            lambda model, blocks: model.diagnose({**blocks, "a": blocks["a"].iloc[:, :4]}),
            ValueError,
            r"^Block 'a' must have 5 columns; got 4\.$",
        ),
        (
            lambda model, blocks: model.spe_contributions(blocks["a"]),
            TypeError,
            r"^X must be a dict\[str, pd\.DataFrame\]\.$",
        ),
        (
            lambda model, blocks: model.spe_contributions({"a": blocks["a"]}),
            ValueError,
            r"^Missing X-blocks: \['b'\]\.$",
        ),
        (
            lambda model, _: model.block_spe_limit("zz"),
            KeyError,
            r"Unknown block 'zz'\. Known blocks: \['a', 'b'\]\.",
        ),
        (
            lambda model, _: model.super_score_plot(pc_horiz=0),
            ValueError,
            r"^pc_horiz and pc_vert must be in 1\.\.2\.$",
        ),
        (
            lambda model, _: model.super_score_plot(pc_vert=3),
            ValueError,
            r"^pc_horiz and pc_vert must be in 1\.\.2\.$",
        ),
    ],
    ids=[
        "diagnose-a-bare-frame",
        "diagnose-a-missing-block",
        "diagnose-a-dropped-column",
        "spe-contributions-of-a-bare-frame",
        "spe-contributions-of-a-missing-block",
        "spe-limit-of-an-unknown-block",
        "score-plot-component-0",
        "score-plot-component-3",
    ],
)
def test_a_fitted_model_refuses_what_it_was_not_fitted_on(
    cls: type[MultiblockModel],
    action: Callable[[MultiblockModel, dict], object],
    error: type[Exception],
    message: str,
) -> None:
    """New data must hold every fitted block at its fitted width, and indices must exist."""
    blocks, y = _blocks()
    model = _fit(cls, blocks, y)
    with pytest.raises(error, match=message):
        action(model, blocks)


@pytest.mark.parametrize(
    ("cls", "method", "component"), [(MBPCA, "super_loadings_bar_plot", 3), (MBPLS, "super_weights_bar_plot", 0)]
)
def test_a_super_bar_plot_refuses_a_component_out_of_range(
    cls: type[MultiblockModel], method: str, component: int
) -> None:
    """The super loading (MBPCA) and super weight (MBPLS) bar plots take a component in 1..A."""
    blocks, y = _blocks()
    with pytest.raises(ValueError, match=r"^component must be in 1\.\.2\.$"):
        getattr(_fit(cls, blocks, y), method)(component)


@pytest.mark.parametrize("cls", [MBPCA, MBPLS])
def test_blocks_given_as_arrays_are_read_with_the_fitted_columns(cls: type[MultiblockModel]) -> None:
    """A block passed as a plain array projects exactly as the same block as a DataFrame."""
    blocks, y = _blocks()
    model = _fit(cls, blocks, y)
    as_arrays = {name: frame.to_numpy() for name, frame in blocks.items()}

    np.testing.assert_allclose(model.diagnose(as_arrays).super_scores, model.diagnose(blocks).super_scores)
    from_arrays, from_frames = model.spe_contributions(as_arrays), model.spe_contributions(blocks)
    for name in blocks:
        np.testing.assert_allclose(from_arrays[name], from_frames[name])
        assert list(from_arrays[name].columns) == list(blocks[name].columns)


@pytest.mark.parametrize("cls", [MBPCA, MBPLS])
def test_spe_limits_follow_the_last_component_spe(cls: type[MultiblockModel]) -> None:
    """Each block's limit is set on its last-component SPE, and the super limit on their root sum of squares."""
    blocks, y = _blocks()
    model = _fit(cls, blocks, y)
    for name in blocks:
        last_spe = model.block_spe_[name].iloc[:, -1].to_numpy()
        assert model.block_spe_limit(name, conf_level=0.95) == pytest.approx(spe_calculation(last_spe, 0.95))
        assert model.block_spe_limit(name, conf_level=0.99) > model.block_spe_limit(name, conf_level=0.95)

    merged = np.sqrt(sum(model.block_spe_[name].iloc[:, -1].to_numpy() ** 2 for name in blocks))
    assert model.super_spe_limit(conf_level=0.95) == pytest.approx(spe_calculation(merged, 0.95))

    # With one block, the merged SPE is that block's SPE, so the two limits agree.
    one_block = _fit(cls, {"a": blocks["a"]}, y)
    assert one_block.super_spe_limit(0.95) == pytest.approx(one_block.block_spe_limit("a", 0.95))


def test_mbpca_with_one_component_reports_its_r2_as_both_increment_and_total() -> None:
    """A one-component MBPCA has the first component of a two-component fit, R2 and all."""
    blocks, _ = _blocks()
    one, two = MBPCA(n_components=1).fit(blocks), MBPCA(n_components=2).fit(blocks)
    pd.testing.assert_frame_equal(one.r2_x_per_block_per_component_, one.r2_x_per_block_cumulative_)
    np.testing.assert_allclose(one.r2_x_per_block_per_component_[1], two.r2_x_per_block_per_component_[1])


def test_mbpca_transform_returns_the_super_scores() -> None:
    """transform() is the Pipeline face of diagnose(): on the training rows it gives the fitted super scores."""
    blocks, _ = _blocks()
    model = MBPCA(n_components=2).fit(blocks)
    scores = model.transform(blocks)
    assert isinstance(scores, pd.DataFrame)
    np.testing.assert_allclose(scores, model.super_scores_, atol=1e-10)


def test_mbpls_accepts_y_as_an_array() -> None:
    """A Y passed as an array fits the same model as the same Y as a DataFrame."""
    blocks, y = _blocks()
    from_frame = MBPLS(n_components=2).fit(blocks, y)
    from_array = MBPLS(n_components=2).fit(blocks, y.to_numpy())
    np.testing.assert_allclose(from_array.predictions_, from_frame.predictions_)


def test_mbpls_refuses_a_y_of_another_height() -> None:
    """Y must have one row per row of the X-blocks."""
    blocks, y = _blocks()
    with pytest.raises(ValueError, match=r"^y has 9 rows; expected 10 to match X-blocks\.$"):
        MBPLS(n_components=2).fit(blocks, y.iloc[:9])


def test_mbpls_super_vip_is_zero_when_y_has_no_variance() -> None:
    """With nothing in Y to explain, R2Y is undefined and the super VIP is zero rather than NaN."""
    blocks, _ = _blocks()
    model = MBPLS(n_components=2).fit(blocks, pd.DataFrame({"constant": np.ones(10)}))
    assert model.r2_y_cumulative_.isna().all()
    np.testing.assert_array_equal(model.super_vip_.to_numpy(), [0.0, 0.0])


def test_mbpls_predictions_plot_defaults_to_the_first_y_and_refuses_an_unknown_one() -> None:
    """Without `variable` the first Y column is drawn; an unknown name lists the known ones."""
    blocks, y = _blocks()
    model = MBPLS(n_components=2).fit(blocks, y)
    fig = model.predictions_vs_observed_plot(y)
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Predicted vs observed for y1"
    np.testing.assert_allclose(fig.data[0].y, model.predictions_["y1"])

    with pytest.raises(ValueError, match=r"^Unknown Y-variable 'nope'\. Known: \['y1', 'y2'\]\.$"):
        model.predictions_vs_observed_plot(y, variable="nope")
