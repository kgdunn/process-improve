# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: process-improve
#     language: python
#     name: python3
# ---

# %%
import os
import pathlib
import sys

import numpy as np
import pandas as pd

# Some basic checks, and logging set up.
while not str(pathlib.Path.cwd().name).lower().startswith("process-improve"):
    os.chdir(pathlib.Path.cwd().parents[0])
basecwd = pathlib.Path.cwd()
sys.path.insert(0, str(basecwd))
assert basecwd.exists()

import process_improve.datasets.batch as batch_ds
from process_improve.batch.data_input import melted_to_dict
from process_improve.batch.plotting import plot_all_batches_per_tag

pd.options.plotting.backend = "plotly"
pd.options.display.max_columns = 20
pd.options.display.width = 200


# %%
# Import the data: a dictionary of dataframes

dryer_raw = pd.read_csv(pathlib.Path(batch_ds.__path__._recalculate()[0]) / "dryer.csv")
df_dict = melted_to_dict(dryer_raw, batch_id_col="batch_id")

# %%
n_levels = 180
percentile_low, percentile_high = 0.05, 0.95  # we create a range of n_levels between this low and high value

# Ideally, use more than 1 tag to align on. These columns must exist in all data frames for all batches.
columns_to_import = ["AgitatorPower", "AgitatorTorque", "JacketTemperature", "DryerTemp"]
tag_to_plot = columns_to_import[2]

# %%
# Plot some data, to get an idea of what is present
plot_all_batches_per_tag(df_dict=df_dict, tag=tag_to_plot, time_column="ClockTime", x_axis_label="Time [hours]")


# %%
for column in columns_to_import:
    series = dryer_raw[column]
    percentiles = series.quantile([percentile_low, percentile_high])

    # Create a range of n_levels between the low and high percentile. then map the data to these levels,
    # but don't clip the values; rather allow them to be outside the range, even if negative.

    bins, step_size = np.linspace(percentiles[percentile_low], percentiles[percentile_high], n_levels + 1, retstep=True)
    # Extend the bins to the left and right to allow for values outside the range, in the same step size
    all_bins = np.concatenate(
        [
            np.flip(np.arange(start=bins[0], stop=series.min() - step_size, step=-step_size))[:-1],
            bins,
            np.arange(start=bins[-1], stop=series.max() + step_size, step=+step_size)[1:],
        ]
    )
    levels = pd.cut(series, bins=all_bins, labels=range(len(all_bins) - 1), include_lowest=True)


# %%
# Next steps: graph storage of the nodes, and then create a graph of the transitions between the nodes.
# Handle pulses.

# %%
