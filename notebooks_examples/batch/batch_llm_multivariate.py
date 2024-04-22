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
# !pip install git+https://github.com/SalesforceAIResearch/uni2ts


from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from gluonts.dataset.pandas import PandasDataset
from huggingface_hub import hf_hub_download
from uni2ts.model.moirai import MoiraiForecast

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


TIME_COL = "Date"
TARGET = "visits"
DYNAMIC_COV = ["CPI", "Inflation_Rate", "GDP"]
SEAS_COV = [
    "month_1",
    "month_2",
    "month_3",
    "month_4",
    "month_5",
    "month_6",
    "month_7",
    "month_8",
    "month_9",
    "month_10",
    "month_11",
    "month_12",
]
FORECAST_HORIZON = 8  # months
FREQ = "M"


# %%
def preprocess_dataset(dataframe: pd.DataFrame, dynamic_cov: list, time_col: str, target: str) -> pd.DataFrame:
    """
    Receives the raw dataframe and create.

        - unique id column
        - make dynamic start dates for each series based on the first date where visits is different than 0

    Args:
        dataframe (pd.DataFrame): raw data
        dynamic_cov (list): column names with dynamic cov
        time_col (str): time column name
        target (str): target name
    Returns:
        pd.DataFrame: cleaned data
    """

    # save dynamic cov for later
    dynamic_cov_df = dataframe[dynamic_cov].reset_index().drop_duplicates()

    # create target and unique id columns
    dataframe = (
        dataframe.loc[:, ~dataframe.columns.isin(dynamic_cov)]
        .melt(ignore_index=False)
        .reset_index()
        .rename(columns={"variable": "unique_id", "value": target})
    )

    # crete dynamic start dates for each series
    cleaned_df = []
    for i in dataframe["unique_id"].unique():
        temp = dataframe[dataframe["unique_id"] == i]
        cleaned_df.append(temp[temp[time_col] >= min(temp[temp[target] > 0][time_col])])
    cleaned_df = pd.concat(cleaned_df)

    # join dynamic cov
    cleaned_df = pd.merge(cleaned_df, dynamic_cov_df, on=[time_col], how="left")

    return cleaned_df


def moirai_forecast_to_pandas(forecast, test_df: pd.DataFrame, forecast_horizon: int, time_col: str) -> pd.DataFrame:
    """
    Convert MOIRAI forecast into pandas dataframe.

    Args:
        forecast: MOIRAI's forecast
        test_df: dataframe with actuals
        forecast_horizon: forecast horizon
        time_col: date column
    Returns:
        pd.DataFrame: forecast in pandas format
    """

    d = {"unique_id": [], time_col: [], "forecast": [], "forecast_lower": [], "forecast_upper": []}

    for ts in forecast:
        for j in range(forecast_horizon):
            d["unique_id"].append(ts.item_id)
            d[time_col].append(test_df[test_df["unique_id"] == ts.item_id][time_col].tolist()[j])

            temp = [sample[j] for sample in ts.samples]

            d["forecast"].append(np.median(temp))
            d["forecast_lower"].append(np.percentile(temp, 10))
            d["forecast_upper"].append(np.percentile(temp, 90))

    return pd.DataFrame(d)


# load data and exogenous features
df = pd.DataFrame(load_dataset("zaai-ai/time_series_datasets", data_files={"train": "data.csv"})["train"]).drop(
    columns=["Unnamed: 0"]
)
df[TIME_COL] = pd.to_datetime(df[TIME_COL])

# one hot encode month
df["month"] = df[TIME_COL].dt.month
df = pd.get_dummies(df, columns=["month"], dtype=int)

print(f"Distinct number of time series: {len(df['unique_id'].unique())}")
df.head()

# 8 months to test
train = df[df[TIME_COL] <= (max(df[TIME_COL]) - pd.DateOffset(months=FORECAST_HORIZON))]
test = df[df[TIME_COL] > (max(df[TIME_COL]) - pd.DateOffset(months=FORECAST_HORIZON))]

print(
    f"Months for training: {len(train[TIME_COL].unique())} from {min(train[TIME_COL]).date()} to {max(train[TIME_COL]).date()}"
)
print(
    f"Months for testing: {len(test[TIME_COL].unique())} from {min(test[TIME_COL]).date()} to {max(test[TIME_COL]).date()}"
)


# create GluonTS dataset from pandas
ds = PandasDataset.from_long_dataframe(
    pd.concat([train, test[["unique_id", TIME_COL] + DYNAMIC_COV + SEAS_COV]]).set_index(
        TIME_COL
    ),  # concatenaation with test dynamic covaraiates
    item_id="unique_id",
    feat_dynamic_real=DYNAMIC_COV + SEAS_COV,
    target=TARGET,
    freq=FREQ,
)


# Prepare pre-trained model by downloading model weights from huggingface hub
model = MoiraiForecast.load_from_checkpoint(
    checkpoint_path=hf_hub_download(repo_id="Salesforce/moirai-R-large", filename="model.ckpt"),
    prediction_length=FORECAST_HORIZON,
    context_length=24,
    patch_size="auto",
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=ds.num_feat_dynamic_real,
    past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    map_location="cuda:0" if torch.cuda.is_available() else "cpu",
)

predictor = model.create_predictor(batch_size=32)
forecasts = predictor.predict(ds)

# convert forecast into pandas
forecast_df = moirai_forecast_to_pandas(forecasts, test, FORECAST_HORIZON, TIME_COL)
