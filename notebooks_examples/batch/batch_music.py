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
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pandas as pd
from neomodel import (
    DateTimeProperty,
    FloatProperty,
    IntegerProperty,
    RelationshipTo,
    StringProperty,
    StructuredNode,
    StructuredRel,
    config,
    db,
    install_all_labels,
)

# Some basic checks, and logging set up.
while not str(pathlib.Path.cwd().name).lower().startswith("process-improve"):
    os.chdir(pathlib.Path.cwd().parents[0])
basecwd = pathlib.Path.cwd()
sys.path.insert(0, str(basecwd))
assert basecwd.exists()

import process_improve.datasets.batch as batch_ds  # noqa: E402
from process_improve.batch.data_input import melted_to_dict  # noqa: E402
from process_improve.batch.plotting import plot_all_batches_per_tag  # noqa: E402

pd.options.plotting.backend = "plotly"
pd.options.display.max_columns = 20
pd.options.display.width = 200
config.AUTO_INSTALL_LABELS = True
config.FORCE_TIMEZONE = True
config.DATABASE_URL = "bolt://neo4j:@localhost:7687"


# Code specific settings
n_levels = 80
percentile_low, percentile_high = 0.05, 0.95  # we create a range of n_levels between this low and high value

# Ideally, use more than 1 tag to align on. These columns must exist in all data frames for all batches.
columns_to_import = ["AgitatorPower", "AgitatorTorque", "JacketTemperature", "DryerTemp"]
tag_to_plot = columns_to_import[2]


# %%
# Import the data: a dictionary of dataframes
dryer_raw = (
    pd.read_csv(pathlib.Path(batch_ds.__path__._recalculate()[0]) / "dryer.csv")
    .rename({"ClockTime": "Pulse"}, axis=1)
    .astype({"Pulse": "int"})
)

# %%
# Set up the graph data model
db.cypher_query("MATCH (n:Pitch) DETACH DELETE n;")


class LeadsTo(StructuredRel):
    """A relationship between two pitches."""

    transition_duration = FloatProperty(required=True)


class Pitch(StructuredNode):
    """A pitch is a single data point in a batch."""

    batch_id = StringProperty(required=True)
    tag = StringProperty(required=True)
    time_since_start = FloatProperty()
    pulse = IntegerProperty()
    datetime = DateTimeProperty()
    pitch_value = IntegerProperty()
    leads_to = RelationshipTo("Pitch", "LEADS_TO", model=LeadsTo)


install_all_labels()


# %%
# Plot some data, to get an idea of what is present
plot_all_batches_per_tag(
    df_dict=melted_to_dict(dryer_raw, batch_id_col="batch_id"),
    tag=tag_to_plot,
    time_column="ClockTime",
    x_axis_label="Time [hours]",
)


# %%
# For all the batches, in one go, create a range of n_levels between the low and high percentile.
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
    # Overwrite the data with the levels
    dryer_raw[column + "_level"] = pd.cut(series, bins=all_bins, labels=range(len(all_bins) - 1), include_lowest=True)


# %%
def store_row(row: pd.Series, batch_id: str, flags: pd.Series | None = None) -> dict:
    """
    Store a row of "pitches" representing batch data.

    An optional flags series can be passed to indicate which tags to store.

    """
    pitches = {}
    for tag_name, pitch_value in row.items():
        tag = tag_name.replace("_level", "")
        if tag == "Pulse":
            continue
        if (flags is not None) and (flags[tag_name] == 0):
            continue
        pitches[tag] = Pitch(
            batch_id=batch_id,
            tag=tag,
            time_since_start=row["Pulse"] * 60 * 60,  # happens to be in hours; convert to seconds
            pulse=int(row["Pulse"]),
            pitch_value=int(pitch_value),
        ).save()
    return pitches


# %%
# Next steps: graph storage of the nodes, and then create a graph of the transitions between the nodes.
# Do this per batch.
df_dict = melted_to_dict(dryer_raw, batch_id_col="batch_id")

for batch_id, df in df_dict.items():
    this_batch = df[["Pulse", *[f"{item}_level" for item in columns_to_import]]]
    # Store the first row
    row_values = this_batch.iloc[0]
    pitches = store_row(row_values, batch_id, flags=None)
    # work with the diffs:
    this_batch_diffs = this_batch.astype("int").diff()
    for _, row in this_batch_diffs[1:].iterrows():  # we've handled row 1 already
        if row.abs().sum() == 1:
            continue  # we don't have any change from the prior row

        row_values += row
        # Create the transitions
        for tag, pitch_value in (next_row := store_row(row_values, batch_id, flags=row)).items():
            pitches[tag].leads_to.connect(
                pitch_value, {"transition_duration": pitch_value.time_since_start - pitches[tag].time_since_start}
            )

        # Update the pitches
        pitches |= next_row
# %%
a = 2

# %%
