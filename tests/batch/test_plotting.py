from process_improve.batch.plotting import (
    plot__all_batches_per_tag,
    plot__tag_time,
    plot_to_HTML,
)
from process_improve.batch.data_input import melt_df_to_series
from process_improve.batch.preprocessing import determine_scaling, apply_scaling


def test_plotting_dryer(dryer_data):
    assert len(dryer_data) == 71
    fig = plot__all_batches_per_tag(
        df_dict=dryer_data,
        tag="JacketTemperature",
        time_column="ClockTime",
        x_axis_label="Samples since start of batch",
    )

    assert len(fig["data"]) == len(dryer_data)


def test_plotting_nylon(nylon_data):
    dict_df = nylon_data
    fig = plot__all_batches_per_tag(
        df_dict=dict_df,
        tag="Tag09",
        x_axis_label="Samples since start of batch",
    )
    assert len(fig["data"]) == len(dict_df)


def test_plotting_tags(nylon_data):
    scale_df = determine_scaling(nylon_data, settings={"robust": False})
    batches_scaled = apply_scaling(nylon_data, scale_df)
    long_form = melt_df_to_series(batches_scaled["1"], name="Raw trajectories")

    fig = plot__tag_time(
        source=long_form,
        overlap=False,
        filled=False,
        # tag_order: Optional[list] = None,
        # x_axis_label: str = "Time, grouped per tag",
    )
    plot_to_HTML("test.html", fig)
    assert len(fig["data"]) == batches_scaled["1"].shape[1] - 1

    # TODO: plot side-by-side and colour filling
