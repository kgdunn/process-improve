import pathlib
from functools import partial

import numpy as np
import pandas as pd
import pytest

from process_improve.batch.plotting import (
    colours_per_batch_id,
    get_rgba_from_triplet,
    plot_all_batches_per_tag,
    plot_multitags,
    plot_to_HTML,
)
from process_improve.batch.preprocessing import apply_scaling, determine_scaling

go = pytest.importorskip("plotly.graph_objects")
sns = pytest.importorskip("seaborn")


def test_plot_colours() -> None:
    """Test colour conversion from triplet to RGBA."""
    assert get_rgba_from_triplet([0.9677975592919913, 0.44127456009157356, 0.5358103155058701]) == pytest.approx(
        [246, 112, 136]
    )

    assert (
        get_rgba_from_triplet(
            [0.9677975592919913, 0.44127456009157356, 0.5358103155058701],
            1,
            as_string=True,
        )
        == "rgba(246,112,136,1.0)"
    )


def test_plotting_dryer(dryer_data: dict) -> None:
    """Test plotting all batches for dryer data."""
    assert len(dryer_data) == 71
    fig = plot_all_batches_per_tag(
        df_dict=dryer_data,
        tag="JacketTemperature",
        time_column="ClockTime",
        x_axis_label="Samples since start of batch",
    )

    assert len(fig["data"]) == len(dryer_data)


def test_plotting_nylon(nylon_data: dict) -> None:
    """Test plotting all batches for nylon data."""
    dict_df = nylon_data
    fig = plot_all_batches_per_tag(
        df_dict=dict_df,
        tag="Tag09",
        tag_y2="Tag07",
        x_axis_label="Samples since start of batch",
        batches_to_highlight={
            '{"width": 4, "color": "rgba(255,0,0,0.5)"}': [2, 3, 4],
            '{"width": 2, "color": "rgba(0,0,255,0.9)"}': [5, 6],
            '{"width": 1, "color": "rgba(255,0,255,0.9)"}': [48],
        },
        y2_limits=(6000, 8000),
    )
    # plot_to_HTML("test.html", fig)
    assert len(fig["data"]) == len(dict_df) * 2  # plotting two tags; double the number.


def test_plotting_nylon_bad_highlight_key_raises_clear_value_error(nylon_data: dict) -> None:
    """SEC-32 (#281): a non-JSON ``batches_to_highlight`` key raises ValueError at
    the API surface, not a confusing ``JSONDecodeError`` from inside a
    comprehension.
    """
    with pytest.raises(ValueError, match="JSON-encoded"):
        plot_all_batches_per_tag(
            df_dict=nylon_data,
            tag="Tag09",
            x_axis_label="Samples since start of batch",
            # Plain string instead of a JSON-encoded line-style spec.
            batches_to_highlight={"not-a-json-key": [2, 3]},
        )


def _two_batches() -> dict:
    """Two four-sample batches with a time column sampled every half unit."""
    return {
        batch_id: pd.DataFrame(
            {"t": np.arange(4.0) * 0.5, "temp": np.arange(4.0) + batch_id, "press": np.arange(4.0) * batch_id}
        )
        for batch_id in (1, 2)
    }


def test_rgba_needs_three_or_four_values() -> None:
    """A colour needs three channels and an optional alpha; two values are refused."""
    with pytest.raises(ValueError, match=r"`incolour` must be a list of 3 or 4 values; got 2 entries\."):
        get_rgba_from_triplet([0.1, 0.2])


def test_plot_to_html_writes_a_standalone_file(tmp_path: pathlib.Path) -> None:
    """The figure is written where asked, loading plotly from the CDN and without the plotly logo."""
    target = tmp_path / "batches.html"
    returned = plot_to_HTML(str(target), plot_all_batches_per_tag(_two_batches(), tag="temp"))
    html = target.read_text()
    assert pathlib.Path(returned) == target
    assert "cdn.plot.ly" in html
    assert '"displaylogo": false' in html


@pytest.mark.parametrize(
    ("tags", "match"),
    [
        ({"tag": "nope"}, r"Tag 'nope' not found in the batch with id 1\."),
        ({"tag": "temp", "tag_y2": "nope"}, r"Tag 'nope' not found in the batch with id 1\."),
    ],
    ids=["left-axis-tag", "right-axis-tag"],
)
def test_plot_all_batches_per_tag_names_a_missing_tag(tags: dict, match: str) -> None:
    """A tag absent from a batch is reported with that batch's identifier."""
    with pytest.raises(KeyError, match=match):
        plot_all_batches_per_tag(_two_batches(), **tags)


def test_plot_all_batches_per_tag_fixes_the_left_axis_range() -> None:
    """Explicit y1 limits switch off autoranging on the left axis."""
    fig = plot_all_batches_per_tag(_two_batches(), tag="temp", y1_limits=(0, 100))
    assert fig.layout.yaxis.range == (0, 100)
    assert fig.layout.yaxis.autorange is False


def test_colours_per_batch_id_defaults_to_the_hls_palette() -> None:
    """Without a colour map the hls palette is used, and a highlighted batch takes its JSON spec."""
    default = colours_per_batch_id([1, 2], {}, 2)
    assert default == colours_per_batch_id([1, 2], {}, 2, colour_map=partial(sns.color_palette, "hls"))
    highlighted = colours_per_batch_id([1, 2], {'{"width": 4, "color": "red"}': [1]}, 2)
    assert highlighted == {1: {"width": 4, "color": "red"}, 2: default[2]}
    with pytest.raises(
        ValueError, match=r"batches_to_highlight: each key must be a JSON-encoded colour spec\. Got 'bad'\."
    ):
        colours_per_batch_id([1, 2], {"bad": [1]}, 2)


def test_plot_multitags_draws_into_a_given_figure_with_the_chosen_layout() -> None:
    """A supplied figure, tag list, batch subset and column count are all honoured; time is not a panel."""
    existing = go.Figure()
    fig = plot_multitags(
        _two_batches(),
        batch_list=[2],
        tag_list=["t", "temp", "press"],
        time_column="t",
        fig=existing,
        settings={"ncols": 2},
    )
    assert fig is existing
    assert [annotation.text for annotation in fig.layout.annotations] == ["temp", "press"]
    assert [trace.name for trace in fig.data] == ["2", "2"]
    np.testing.assert_array_equal(fig.data[0].x, [0.0, 0.5, 1.0, 1.5])


def test_plot_multitags_highlights_batches_by_their_spec() -> None:
    """Every trace of a highlighted batch carries its JSON line spec."""
    fig = plot_multitags(_two_batches(), batches_to_highlight={'{"width": 4, "color": "red"}': [1]})
    highlighted = [(trace.line.width, trace.line.color) for trace in fig.data if trace.name == "1"]
    assert highlighted == [(4, "red")] * 3


def test_plot_multitags_animation_follows_the_time_column_and_can_drop_pause() -> None:
    """Animated frames use the time column on the x-axis, and the pause button is optional."""
    fig = plot_multitags(
        _two_batches(),
        time_column="t",
        settings={
            "animate": True,
            "animate_batches_to_highlight": [1],
            "animate_n_frames": 2,
            "animate_show_pause": False,
        },
    )
    assert [button.label for button in fig.layout.updatemenus[0].buttons] == ["Play"]
    np.testing.assert_array_equal(fig.frames[-1].data[0].x, [0.0, 0.5, 1.0, 1.5])


def test_plot_multitags_rejects_a_non_numeric_tag() -> None:
    """The tags to plot are validated across the batches, and the failing batch is named."""
    batches = {
        1: pd.DataFrame({"temp": [10.0, 11.0], "operator": ["ann", "bob"]}),
        2: pd.DataFrame({"temp": [12.0, 13.0], "operator": ["cy", "di"]}),
    }
    with pytest.raises(ValueError, match=r"All columns must be a numeric type\. Differs in 1\."):
        plot_multitags(df_dict=batches)


def test_plotting_tags(nylon_data: dict) -> None:
    """Test plotting multiple tags."""
    scale_df = determine_scaling(nylon_data, settings={"robust": False})
    batches_scaled = apply_scaling(nylon_data, scale_df)

    fig = plot_multitags(df_dict=batches_scaled)
    assert len(fig["data"]) == len(batches_scaled) * batches_scaled[1].shape[1]


def test_plot_all_batches_per_tag_mode(dryer_data: dict) -> None:
    """The `mode` argument should propagate to every Plotly trace."""
    fig = plot_all_batches_per_tag(
        df_dict=dryer_data,
        tag="JacketTemperature",
        time_column="ClockTime",
        mode="lines+markers",
    )
    assert all(trace["mode"] == "lines+markers" for trace in fig["data"])


def test_plot_multitags_mode(nylon_data: dict) -> None:
    """The `mode` setting should propagate to every Plotly trace."""
    scale_df = determine_scaling(nylon_data, settings={"robust": False})
    batches_scaled = apply_scaling(nylon_data, scale_df)

    fig = plot_multitags(df_dict=batches_scaled, settings={"mode": "lines+markers"})
    assert all(trace["mode"] == "lines+markers" for trace in fig["data"])


def test_plot_multitags_pause_button_targets_current_animation(nylon_data: dict) -> None:
    """Test the animation pause button uses Plotly's current animation sentinel."""
    scale_df = determine_scaling(nylon_data, settings={"robust": False})
    batches_scaled = apply_scaling(nylon_data, scale_df)

    fig = plot_multitags(
        df_dict=batches_scaled,
        settings={
            "animate": True,
            "animate_batches_to_highlight": [1],
            "animate_n_frames": 2,
        },
    )

    buttons = fig.layout.updatemenus[0].buttons
    pause_button = next(button for button in buttons if button.label == "Pause")
    assert pause_button.args[0] == (None,)
