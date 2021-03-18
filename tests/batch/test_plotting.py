from process_improve.batch.plotting import plot__all_batches_per_tag  # plot_to_HTML


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
