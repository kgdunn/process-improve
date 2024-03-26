import pytest

from process_improve.batch.plotting import (
    get_rgba_from_triplet,
    plot_all_batches_per_tag,
    plot_multitags,
)
from process_improve.batch.preprocessing import apply_scaling, determine_scaling


def test_plot_colours():
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


def test_plotting_dryer(dryer_data):
    assert len(dryer_data) == 71
    fig = plot_all_batches_per_tag(
        df_dict=dryer_data,
        tag="JacketTemperature",
        time_column="ClockTime",
        x_axis_label="Samples since start of batch",
    )

    assert len(fig["data"]) == len(dryer_data)


def test_plotting_nylon(nylon_data):
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


def test_plotting_tags(nylon_data):
    scale_df = determine_scaling(nylon_data, settings={"robust": False})
    batches_scaled = apply_scaling(nylon_data, scale_df)

    fig = plot_multitags(df_dict=batches_scaled)
    assert len(fig["data"]) == len(batches_scaled) * batches_scaled[1].shape[1]
