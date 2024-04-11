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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from chronos import ChronosPipeline

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

device_map = "cpu"  # use "cpu" for CPU inference and "mps" for Apple Silicon; or "cuda"

# Ideally, use more than 1 tag to align on. These columns must exist in all data frames for all batches.
columns_to_import = ["AgitatorPower", "AgitatorTorque", "JacketTemperature", "DryerTemp"]
tag_to_plot = columns_to_import[3]


# %%
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-base",
    device_map=device_map,
    torch_dtype=torch.bfloat16,
)

# %%
df = pd.read_csv(
    "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
)
df["#Passengers"].plot()

# %%

# %%
# context must be either a 1D tensor, a list of 1D tensors,
# or a left-padded 2D tensor with batch as the first dimension
context = torch.tensor(df["#Passengers"])
prediction_length = 12
forecast = pipeline.predict(
    context,
    prediction_length,
    num_samples=20,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
)  # forecast shape: [num_series, num_samples, prediction_length]
forecast

# %%
# visualize the forecast
forecast_index = range(len(df), len(df) + prediction_length)


# %%

low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)

# %%
plt.figure(figsize=(8, 4))
plt.plot(df["#Passengers"], color="royalblue", label="historical data")
plt.plot(forecast_index, median, color="tomato", label="median forecast")
plt.fill_between(forecast_index, low, high, color="tomato", alpha=0.3, label="80% prediction interval")
plt.legend()
plt.grid()
plt.show()

# %%
context = torch.tensor(df["#Passengers"])
embeddings, tokenizer_state = pipeline.embed(context)
embeddings

# %%
# Import the data: a dictionary of dataframes
dryer_raw = (
    pd.read_csv(pathlib.Path(batch_ds.__path__._recalculate()[0]) / "dryer.csv")
    .rename({"ClockTime": "Pulse"}, axis=1)
    .astype({"Pulse": "int"})
)
dryer_raw[tag_to_plot].plot()

# %%
# Plot some data, to get an idea of what is present
plot_all_batches_per_tag(
    df_dict=melted_to_dict(dryer_raw, batch_id_col="batch_id"),
    tag=tag_to_plot,
    time_column="ClockTime",
    x_axis_label="Time [hours]",
)


# %%
training_batches = list(range(1, 10))
batch_dict = melted_to_dict(dryer_raw, batch_id_col="batch_id")
sequences = [batch_dict[k][tag_to_plot].to_list() for k in training_batches if tag_to_plot in batch_dict[k]]
training_sequence = pd.DataFrame(sequences).T
training_sequence.plot(title="Training data")

# %%
# Unravel all columnes, and make a single vector sequence
all_sequences = []
for k in training_batches:
    all_sequences.extend(batch_dict[k][tag_to_plot].to_list())
training_sequence = pd.Series(all_sequences)
training_sequence.plot(title="Unravelled training data")

# %%
# Train the model with the tag from the training batches
context = torch.tensor(training_sequence)
prediction_length = 12
forecast = pipeline.predict(
    context,
    prediction_length,
    num_samples=20,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
)  # forecast shape: [num_series, num_samples, prediction_length]

# %%
# Testing data: the first 50 samples of the next 10 batches
first_n_samples = 70
testing_batches = list(range(10, 20))
batch_dict = melted_to_dict(dryer_raw, batch_id_col="batch_id")
sequences = [batch_dict[k][tag_to_plot].to_list()[0:first_n_samples] for k in testing_batches]
testing_sequence = pd.DataFrame(sequences).T
testing_sequence.plot(title="Testing data").show()

# %%
# Testing data: the first 50 samples of the next 10 batches
prediction_length = 25
for batch in testing_batches:
    context = torch.tensor(
        pd.concat([training_sequence, batch_dict[batch].iloc[0:first_n_samples][tag_to_plot]]).values
    )
    cast = pipeline.predict(
        context,
        prediction_length,
        num_samples=20,
        temperature=1,
        top_k=50,
        top_p=1.0,
    )  # forecast shape: [num_series, num_samples, prediction_length]
    lower, median, upper = np.quantile(cast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
    actual = batch_dict[batch][tag_to_plot]
    forecast_index = actual.index[first_n_samples : (first_n_samples + prediction_length)]
    fig = actual.plot(title="Actual and Forecast")
    fig.add_scatter(x=forecast_index, y=median, mode="lines", name="Forecast")
    fill_colour = "rgba(1, 0, 0, 0.2)"
    fig.add_traces(
        go.Scatter(
            x=forecast_index,
            y=upper,
            line=dict(color=fill_colour),
            mode="lines",
        )
    )
    fig.add_traces(
        go.Scatter(
            x=forecast_index,
            y=lower,
            line=dict(color=fill_colour),
            fill="tonexty",  # fill in the range
            fillcolor=fill_colour,
            mode="lines",
        )
    )

    fig.show()


# %%
# batch_dict[batch].iloc[0:first_n_samples][], tag_to_plot]
