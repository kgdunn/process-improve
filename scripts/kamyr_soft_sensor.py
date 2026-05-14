"""
Soft-sensor case study for pid-book Chapter 7.3.

Build a PLS model that predicts Kappa number (lab) from real-time process tags
on a Kamyr digester, using the full openmv.net Kamyr digester data
(301 hourly samples, 22 process tags + Y-Kappa).

Generates the four PNG figures referenced by `soft-sensors.rst` into the
sister `figures` repository (`figures/monitoring/`).
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from process_improve.multivariate import PLS, MCUVScaler

FIGURES_DIR = pathlib.Path("/home/user/figures/monitoring")
DATA_PATH = pathlib.Path("/tmp/kamyr_full.csv")  # the openmv full CSV

Y_COL = "Y-Kappa"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=["Observation"])
    # Drop the two columns with ~47% missing values.
    df = df.drop(columns=["AAWhiteSt-4", "SulphidityL-4"])
    # Replace remaining NaN with column median.
    df = df.fillna(df.median(numeric_only=True))
    return df


def fit_pls(df: pd.DataFrame, n_components: int = 2) -> dict:
    x_cols = [c for c in df.columns if c != Y_COL]
    X = df[x_cols].copy()
    y = df[[Y_COL]].copy()
    scaler_x = MCUVScaler().fit(X)
    scaler_y = MCUVScaler().fit(y)
    model = PLS(n_components=n_components).fit(scaler_x.transform(X), scaler_y.transform(y))
    return {"model": model, "X": X, "y": y, "x_cols": x_cols}


def evaluate_split(df: pd.DataFrame, x_cols: list[str], y_col: str, frac: float = 0.70) -> dict:
    n_train = int(round(frac * len(df)))
    train, test = df.iloc[:n_train].copy(), df.iloc[n_train:].copy()
    X_tr, y_tr = train[x_cols], train[[y_col]]
    X_te, y_te = test[x_cols], test[[y_col]]
    scaler_x = MCUVScaler().fit(X_tr)
    scaler_y = MCUVScaler().fit(y_tr)
    model = PLS(n_components=2).fit(scaler_x.transform(X_tr), scaler_y.transform(y_tr))
    result = model.predict(scaler_x.transform(X_te))
    y_te_hat_mcuv = pd.DataFrame(np.asarray(result.y_hat), index=X_te.index, columns=[y_col])
    y_te_hat = scaler_y.inverse_transform(y_te_hat_mcuv).values.ravel()
    rmsep = float(np.sqrt(np.mean((y_te.values.ravel() - y_te_hat) ** 2)))
    r2_y_train = float(model.r2_cumulative_.iloc[-1])
    return {
        "model": model,
        "rmsep": rmsep,
        "r2_y_train": r2_y_train,
        "y_te_hat": y_te_hat,
        "y_te": y_te.values.ravel(),
        "n_train": len(train),
        "n_test": len(test),
    }


def figure_raw_data(df: pd.DataFrame, out: pathlib.Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    sample = np.arange(len(df))
    axes[0].plot(sample, df[Y_COL], "k-", linewidth=1.0)
    axes[0].set_ylabel("Y-Kappa")
    axes[0].set_title("Kamyr digester: Y-Kappa and two of the pre-shifted process tags")
    axes[1].plot(sample, df["ChipLevel4"], "C0-", linewidth=1.0)
    axes[1].set_ylabel("ChipLevel4")
    axes[2].plot(sample, df["BlackFlow-2"], "C3-", linewidth=1.0)
    axes[2].set_ylabel("BlackFlow-2")
    axes[2].set_xlabel("Sample (1 hour spacing)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_coefficients(model, x_cols, out: pathlib.Path) -> None:
    coefs = model.beta_coefficients_.iloc[:, 0]
    names = list(coefs.index)
    values = coefs.values
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(names, values, color=["C0" if c >= 0 else "C3" for c in values])
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_ylabel("Coefficient on scaled X")
    ax.set_title("PLS regression coefficients onto Y-Kappa (2 components)")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_obs_pred(y: np.ndarray, y_hat: np.ndarray, title: str, out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    lo, hi = float(np.min([y, y_hat])), float(np.max([y, y_hat]))
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=0.8, label="ideal")
    ax.plot(y, y_hat, "o", markersize=6, alpha=0.8)
    ax.set_xlabel("Observed Kappa")
    ax.set_ylabel("Predicted Kappa")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_time_series(
    y_obs: np.ndarray, y_hat_base: np.ndarray, y_hat_lag: np.ndarray, out: pathlib.Path
) -> None:
    """Overlay the actual Kappa with both model predictions on the held-out test set."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    sample = np.arange(len(y_obs))
    ax.plot(sample, y_obs, color="black", linewidth=1.8, label="Lab (actual)")
    ax.plot(
        sample,
        y_hat_base,
        color="C0",
        linestyle="--",
        linewidth=1.4,
        marker="o",
        markersize=4,
        label="Soft sensor: process tags only",
    )
    ax.plot(
        sample[-len(y_hat_lag) :],
        y_hat_lag,
        color="C3",
        linestyle=":",
        linewidth=1.4,
        marker="s",
        markersize=4,
        label="Soft sensor: process tags + Kappa lag",
    )
    ax.set_xlabel("Test sample index (1 hour spacing)")
    ax.set_ylabel("Kappa number")
    ax.set_title("Held-out test predictions vs lab values")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    print(f"Rows: {len(df)} | Columns (after drop): {len(df.columns)}")
    print(f"X tags: {[c for c in df.columns if c != Y_COL]}")
    print()

    # Baseline PLS on full data --------------------------------------------------
    base = fit_pls(df)
    r2_y = base["model"].r2_cumulative_.values
    print(f"Baseline R2_Y cumulative (per component): {r2_y}")

    figure_raw_data(df, FIGURES_DIR / "Kappa-soft-sensor-raw-data.png")
    figure_coefficients(base["model"], base["x_cols"], FIGURES_DIR / "Kappa-soft-sensor-coefficients.png")

    # 70/30 train/test split for RMSEP ------------------------------------------
    base_split = evaluate_split(df, base["x_cols"], Y_COL, frac=0.70)
    print(f"Train rows: {base_split['n_train']}, Test rows: {base_split['n_test']}")
    print(f"Baseline R2_Y(train): {base_split['r2_y_train']:.3f}, RMSEP(test): {base_split['rmsep']:.3f}")
    figure_obs_pred(
        base_split["y_te"],
        base_split["y_te_hat"],
        "Soft sensor predictions on held-out test set (no lag of Y)",
        FIGURES_DIR / "Kappa-soft-sensor-obs-pred-base.png",
    )

    # Lag-augmented model: previous Kappa as an extra X column ------------------
    df_lag = df.copy()
    df_lag["Kappa_lag1"] = df_lag[Y_COL].shift(1)
    df_lag = df_lag.dropna(subset=["Kappa_lag1"]).reset_index(drop=True)
    x_cols_lag = base["x_cols"] + ["Kappa_lag1"]
    lag_split = evaluate_split(df_lag, x_cols_lag, Y_COL, frac=0.70)
    print(f"Lag-augmented R2_Y(train): {lag_split['r2_y_train']:.3f}, RMSEP(test): {lag_split['rmsep']:.3f}")
    figure_obs_pred(
        lag_split["y_te"],
        lag_split["y_te_hat"],
        "Soft sensor predictions with 1-step Kappa lag added to X",
        FIGURES_DIR / "Kappa-soft-sensor-obs-pred-lagged.png",
    )

    # Time-series overlay of test predictions vs actual ----------------------
    figure_time_series(
        base_split["y_te"],
        base_split["y_te_hat"],
        lag_split["y_te_hat"],
        FIGURES_DIR / "Kappa-soft-sensor-time-series.png",
    )

    print()
    print("Wrote five PNG figures to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
