"""(c) Kevin Dunn, 2010-2026. MIT License.

Agent-callable tool wrappers for multivariate analysis (PCA, PLS, scaling).

Pydantic input contract (ENG-04 / ENG-10): each tool pairs its
``@tool_spec`` decorator with a ``BaseModel`` carrying
``ConfigDict(extra="forbid")``; the function receives the parsed
model as its single positional argument.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from process_improve.config import settings
from process_improve.tool_spec import clean, get_tool_specs, tool_spec


def _validate_matrix_shape(data: list, name: str) -> None:
    """Reject oversize matrix inputs at the MCP boundary (SEC-19 #268).

    Caps ``len(data)`` against ``settings.max_matrix_rows`` and the
    first row's width against ``settings.max_matrix_cols``. A 1 x 1M
    matrix passes the ``max_cells`` budget but blows up SVD; this is
    the dimension-aware guard.
    """
    n_rows = len(data)
    if n_rows > settings.max_matrix_rows:
        raise ValueError(
            f"{name} has {n_rows} rows; the cap is "
            f"settings.max_matrix_rows={settings.max_matrix_rows}."
        )
    if n_rows == 0:
        return
    n_cols = len(data[0]) if hasattr(data[0], "__len__") else 1
    if n_cols > settings.max_matrix_cols:
        raise ValueError(
            f"{name} has {n_cols} columns; the cap is "
            f"settings.max_matrix_cols={settings.max_matrix_cols}."
        )


def _validate_model_params(  # noqa: C901
    model_params: dict,
    *,
    keys_2d: tuple[str, ...],
    keys_1d: tuple[str, ...],
    scalar_keys: tuple[str, ...] = ("n_components", "n_samples"),
) -> None:
    """Cap every array-like sub-field of a ``model_params`` payload (SEC-25 #274).

    ``pca_predict`` / ``pls_predict`` previously read attacker-controlled
    sizes straight into ``np.array(...)`` and could be allocation-bombed
    with a multi-million-element nested list (the input ``model_params``
    field had no inner schema, no ``properties``, no ``maxItems``).

    This guard runs **before** any ``np.array(...)`` allocation:

    - 2-D keys (loadings, weights, beta-coefficients) are checked via
      :func:`_validate_matrix_shape`, which caps rows against
      ``settings.max_matrix_rows`` and cols against
      ``settings.max_matrix_cols``.
    - 1-D keys (means, stds, spe_values, scaling factors) are length-
      capped against ``settings.max_matrix_rows``.
    - Scalar keys (``n_components``, ``n_samples``) are capped against
      ``settings.max_matrix_cols`` and ``settings.max_matrix_rows``
      respectively; non-integer values are rejected outright.

    Missing keys propagate as the natural ``KeyError`` from the caller's
    own dict access; the goal of this function is solely to bound the
    sizes, not to enforce the full schema (that lives on the underlying
    fit_pca / fit_pls output).
    """
    for key in keys_2d:
        value = model_params.get(key)
        if value is not None:
            _validate_matrix_shape(value, f"model_params[{key!r}]")

    for key in keys_1d:
        value = model_params.get(key)
        if value is None:
            continue
        n = len(value)
        if n > settings.max_matrix_rows:
            raise ValueError(
                f"model_params[{key!r}] has {n} entries; the cap is "
                f"settings.max_matrix_rows={settings.max_matrix_rows}."
            )

    for key in scalar_keys:
        if key not in model_params:
            continue
        value = model_params[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"model_params[{key!r}] must be an int, got {type(value).__name__}."
            )
        cap = settings.max_matrix_cols if key == "n_components" else settings.max_matrix_rows
        if value > cap:
            raise ValueError(
                f"model_params[{key!r}]={value} exceeds the cap of {cap}."
            )
        if value < 0:
            raise ValueError(
                f"model_params[{key!r}]={value} must be non-negative."
            )


_MULTIVARIATE_TOOL_NAMES: list[str] = []


def _register(name: str) -> None:
    _MULTIVARIATE_TOOL_NAMES.append(name)


# ---------------------------------------------------------------------------
# fit_pca
# ---------------------------------------------------------------------------


class FitPcaInput(BaseModel):
    """Input contract for ``fit_pca``."""

    model_config = ConfigDict(extra="forbid")

    data: list[list[float]] = Field(
        ...,
        min_length=3,
        description="Data matrix as a list of rows, where each row is a list of numeric values.",
    )
    n_components: int = Field(
        ...,
        ge=1,
        description="Number of principal components to extract.",
    )
    column_names: list[str] | None = Field(
        None,
        description="Optional names for each column/variable. Length must match the number of columns.",
    )


@tool_spec(
    name="fit_pca",
    description=(
        "Fit a Principal Component Analysis (PCA) model to a numeric data matrix. "
        "The data is automatically mean-centred and scaled to unit variance (MCUV) before fitting. "
        "Returns explained variance, R-squared statistics, detected outliers, and serialised "
        "model parameters that can be passed to pca_predict for scoring new observations. "
        "Use this when you want to reduce dimensionality, identify patterns, or detect outliers "
        "in a multivariate dataset."
    ),
    input_model=FitPcaInput,
    examples="""
    # "Run PCA with 2 components on my data"
        -> ``fit_pca(data=[[1,2,3],[4,5,6],[7,8,9]], n_components=2)``

    # "Fit a 3-component PCA with named variables"
        -> ``fit_pca(data=[[...],[...]], n_components=3, column_names=["temp","pressure","flow"])``
    """,
    category="multivariate",
)
def fit_pca(spec: FitPcaInput) -> dict[str, Any]:
    """Fit a PCA model to the given data."""
    try:
        _validate_matrix_shape(spec.data, "data")
        from process_improve.multivariate.methods import PCA, MCUVScaler  # noqa: PLC0415

        df = pd.DataFrame(spec.data, columns=spec.column_names)
        scaler = MCUVScaler().fit(df)
        X_scaled = scaler.transform(df)

        model = PCA(n_components=spec.n_components).fit(X_scaled)

        outliers = model.detect_outliers(conf_level=0.95)

        model_params = {
            "loadings": model.loadings_.values.tolist(),
            "means": scaler.center_.values.tolist(),
            "stds": scaler.scale_.values.tolist(),
            "n_components": spec.n_components,
            "scaling_factor_for_scores": model.scaling_factor_for_scores_.values.tolist(),
            "n_samples": int(model.n_samples_),
            "spe_values": model.spe_.iloc[:, -1].values.tolist(),
        }

        result: dict[str, Any] = {
            "n_components": spec.n_components,
            "n_samples": int(model.n_samples_),
            "n_features": int(model.n_features_in_),
            "explained_variance": model.explained_variance_.tolist(),
            "r2_cumulative": model.r2_cumulative_.values.tolist(),
            "r2_per_component": model.r2_per_component_.values.tolist(),
            "outlier_indices": [o["observation"] for o in outliers],
            "outlier_details": outliers,
            "model_params": model_params,
        }
        return clean(result)
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
        return {"error": str(exc)}


_register("fit_pca")


# ---------------------------------------------------------------------------
# fit_pls
# ---------------------------------------------------------------------------


class FitPlsInput(BaseModel):
    """Input contract for ``fit_pls``."""

    model_config = ConfigDict(extra="forbid")

    x_data: list[list[float]] = Field(
        ...,
        min_length=3,
        description="X data matrix as a list of rows.",
    )
    y_data: list[list[float]] | list[float] = Field(
        ...,
        min_length=3,
        description=(
            "Y data as a list of lists (multiple responses) or a flat list of numbers "
            "(single response)."
        ),
    )
    n_components: int = Field(
        ...,
        ge=1,
        description="Number of latent components to extract.",
    )
    x_column_names: list[str] | None = Field(
        None,
        description="Optional names for X columns.",
    )
    y_column_names: list[str] | None = Field(
        None,
        description="Optional names for Y columns.",
    )


@tool_spec(
    name="fit_pls",
    description=(
        "Fit a Projection to Latent Structures (PLS) regression model linking an X matrix to "
        "a Y matrix (or vector). The data is automatically mean-centred and scaled to unit "
        "variance (MCUV) before fitting. "
        "Returns R-squared statistics, RMSE, regression coefficients, and serialised model "
        "parameters that can be passed to pls_predict for scoring new observations. "
        "Use this when you want to build a predictive model from a multivariate X to one or "
        "more Y responses."
    ),
    input_model=FitPlsInput,
    examples="""
    # "Build a PLS model with 2 components predicting yield from process variables"
        -> ``fit_pls(x_data=[[...],[...]], y_data=[10.1, 10.5, ...], n_components=2)``

    # "Fit PLS with named X and Y variables"
        -> ``fit_pls(x_data=[[...]], y_data=[[...]], n_components=3,
                     x_column_names=["temp","flow"], y_column_names=["yield"])``
    """,
    category="multivariate",
)
def fit_pls(spec: FitPlsInput) -> dict[str, Any]:
    """Fit a PLS model to X and Y data."""
    try:
        _validate_matrix_shape(spec.x_data, "x_data")
        if spec.y_data and hasattr(spec.y_data[0], "__len__"):
            _validate_matrix_shape(spec.y_data, "y_data")
        from process_improve.multivariate.methods import PLS, MCUVScaler  # noqa: PLC0415

        X = pd.DataFrame(spec.x_data, columns=spec.x_column_names)

        if spec.y_data and not isinstance(spec.y_data[0], list):
            Y = pd.DataFrame(spec.y_data, columns=spec.y_column_names or ["y"])
        else:
            Y = pd.DataFrame(spec.y_data, columns=spec.y_column_names)

        scaler_x = MCUVScaler().fit(X)
        scaler_y = MCUVScaler().fit(Y)
        X_scaled = scaler_x.transform(X)
        Y_scaled = scaler_y.transform(Y)

        model = PLS(n_components=spec.n_components, scale=False).fit(X_scaled, Y_scaled)

        model_params = {
            "x_loadings": model.x_loadings_.values.tolist(),
            "y_loadings": cast("pd.DataFrame", model.y_loadings_).values.tolist(),
            "direct_weights": model.direct_weights_.values.tolist(),
            "beta_coefficients": model.beta_coefficients_.values.tolist(),
            "x_means": scaler_x.center_.values.tolist(),
            "x_stds": scaler_x.scale_.values.tolist(),
            "y_means": scaler_y.center_.values.tolist(),
            "y_stds": scaler_y.scale_.values.tolist(),
            "n_components": spec.n_components,
            "scaling_factor_for_scores": model.scaling_factor_for_scores_.values.tolist(),
            "n_samples": int(model.n_samples_),
            "spe_values": model.spe_.iloc[:, -1].values.tolist(),
        }

        result: dict[str, Any] = {
            "n_components": spec.n_components,
            "r2x_cumulative": model.r2_cumulative_.values.tolist(),
            "r2_per_component": model.r2_per_component_.values.tolist(),
            "rmse": model.rmse_.values.tolist(),
            "coefficients": model.beta_coefficients_.values.tolist(),
            "model_params": model_params,
        }
        return clean(result)
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
        return {"error": str(exc)}


_register("fit_pls")


# ---------------------------------------------------------------------------
# scale_data
# ---------------------------------------------------------------------------


class ScaleDataInput(BaseModel):
    """Input contract for ``scale_data``."""

    model_config = ConfigDict(extra="forbid")

    data: list[list[float]] = Field(
        ...,
        min_length=1,
        description="Data matrix as a list of rows.",
    )
    column_names: list[str] | None = Field(
        None,
        description="Optional names for each column/variable.",
    )


@tool_spec(
    name="scale_data",
    description=(
        "Mean-centre and scale a data matrix to unit variance (MCUV scaling). "
        "Returns the scaled data along with the column means and standard deviations used. "
        "This is the standard preprocessing step before PCA or PLS. "
        "Use this when you need the scaled data for further analysis or want to inspect "
        "the scaling parameters."
    ),
    input_model=ScaleDataInput,
    examples="""
    # "Scale my data to zero mean and unit variance"
        -> ``scale_data(data=[[1,2],[3,4],[5,6]])``

    # "Scale with named columns"
        -> ``scale_data(data=[[10,20],[30,40]], column_names=["temp","pressure"])``
    """,
    category="multivariate",
)
def scale_data(spec: ScaleDataInput) -> dict[str, Any]:
    """Mean-centre and scale data to unit variance."""
    try:
        _validate_matrix_shape(spec.data, "data")
        from process_improve.multivariate.methods import MCUVScaler  # noqa: PLC0415

        df = pd.DataFrame(spec.data, columns=spec.column_names)
        scaler = MCUVScaler().fit(df)
        scaled = scaler.transform(df)

        result: dict[str, Any] = {
            "scaled_data": scaled.values.tolist(),
            "means": scaler.center_.values.tolist(),
            "stds": scaler.scale_.values.tolist(),
        }
        return clean(result)
    except (ValueError, TypeError, KeyError) as exc:
        return {"error": str(exc)}


_register("scale_data")


# ---------------------------------------------------------------------------
# detect_multivariate_outliers
# ---------------------------------------------------------------------------


class DetectMultivariateOutliersInput(BaseModel):
    """Input contract for ``detect_multivariate_outliers``."""

    model_config = ConfigDict(extra="forbid")

    data: list[list[float]] = Field(
        ...,
        min_length=3,
        description="Data matrix as a list of rows.",
    )
    n_components: int = Field(
        ...,
        ge=1,
        description="Number of PCA components to use for outlier detection.",
    )
    conf_level: float = Field(
        0.95,
        ge=0.8,
        le=0.999,
        description="Confidence level for the statistical limits (default 0.95).",
    )


@tool_spec(
    name="detect_multivariate_outliers",
    description=(
        "Detect multivariate outliers by fitting a PCA model and using both SPE (Squared "
        "Prediction Error) and Hotelling's T-squared diagnostics. "
        "Combines statistical limits with the robust generalised ESD test. "
        "The data is automatically mean-centred and scaled before fitting. "
        "Use this when you suspect some observations are unusual in a multivariate sense "
        "even if they appear normal in individual variables."
    ),
    input_model=DetectMultivariateOutliersInput,
    examples="""
    # "Are there multivariate outliers in my data?"
        -> ``detect_multivariate_outliers(data=[[...],[...]], n_components=2)``

    # "Detect outliers at 99% confidence"
        -> ``detect_multivariate_outliers(data=[[...]], n_components=2, conf_level=0.99)``
    """,
    category="multivariate",
)
def detect_multivariate_outliers(spec: DetectMultivariateOutliersInput) -> dict[str, Any]:
    """Detect multivariate outliers via PCA diagnostics."""
    try:
        _validate_matrix_shape(spec.data, "data")
        from process_improve.multivariate.methods import PCA, MCUVScaler  # noqa: PLC0415

        df = pd.DataFrame(spec.data)
        scaler = MCUVScaler().fit(df)
        X_scaled = scaler.transform(df)

        model = PCA(n_components=spec.n_components).fit(X_scaled)
        outliers = model.detect_outliers(conf_level=spec.conf_level)

        t2_lim = float(model.hotellings_t2_limit(conf_level=spec.conf_level))
        spe_lim = float(model.spe_limit(conf_level=spec.conf_level))

        result: dict[str, Any] = {
            "outlier_indices": [o["observation"] for o in outliers],
            "outlier_details": outliers,
            "t2_limit": t2_lim,
            "spe_limit": spe_lim,
        }
        return clean(result)
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
        return {"error": str(exc)}


_register("detect_multivariate_outliers")


# ---------------------------------------------------------------------------
# pca_predict
# ---------------------------------------------------------------------------


class PcaPredictInput(BaseModel):
    """Input contract for ``pca_predict``."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_params: dict[str, Any] = Field(
        ...,
        description="The model_params dict returned by fit_pca.",
    )
    new_data: list[list[float]] = Field(
        ...,
        min_length=1,
        description="New data matrix as a list of rows (same number of columns as training data).",
    )


@tool_spec(
    name="pca_predict",
    description=(
        "Project new observations into an existing PCA model and compute diagnostics. "
        "Accepts the model_params dict returned by fit_pca and new data rows. "
        "Returns scores, Hotelling's T-squared, SPE, and whether each observation is an "
        "outlier (T-squared exceeding the training-data-based limit). "
        "Use this to monitor new data against a PCA model built on historical data."
    ),
    input_model=PcaPredictInput,
    examples="""
    # "Score new observations against my PCA model"
        -> ``pca_predict(model_params=<from fit_pca>, new_data=[[1,2,3],[4,5,6]])``
    """,
    category="multivariate",
)
def pca_predict(spec: PcaPredictInput) -> dict[str, Any]:
    """Project new data into a PCA model."""
    try:
        _validate_matrix_shape(spec.new_data, "new_data")
        # SEC-25 (#274): cap every array-like sub-field of model_params
        # before any np.array(...) allocation.
        _validate_model_params(
            spec.model_params,
            keys_2d=("loadings",),
            keys_1d=("means", "stds", "scaling_factor_for_scores", "spe_values"),
        )
        from process_improve.multivariate.methods import hotellings_t2_limit, spe_calculation  # noqa: PLC0415

        means = np.array(spec.model_params["means"])
        stds = np.array(spec.model_params["stds"])
        loadings = np.array(spec.model_params["loadings"])
        scaling_factors = np.array(spec.model_params["scaling_factor_for_scores"])
        n_components = spec.model_params["n_components"]
        n_samples = spec.model_params["n_samples"]
        train_spe_values = np.array(spec.model_params["spe_values"])

        X_new = np.array(spec.new_data, dtype=float)
        X_scaled = (X_new - means) / stds

        scores = X_scaled @ loadings
        t2_values = np.sum((scores / scaling_factors) ** 2, axis=1)

        X_hat = scores @ loadings.T
        residuals = X_scaled - X_hat
        spe_values = np.sqrt(np.sum(residuals**2, axis=1))

        t2_lim = hotellings_t2_limit(
            conf_level=0.95,
            n_components=n_components,
            n_rows=n_samples,
        )
        spe_lim = spe_calculation(spe_values=train_spe_values, conf_level=0.95)

        is_outlier = [
            (bool(t2 > t2_lim) or bool(spe > spe_lim))
            for t2, spe in zip(t2_values, spe_values, strict=False)
        ]

        result: dict[str, Any] = {
            "scores": scores.tolist(),
            "hotellings_t2": t2_values.tolist(),
            "spe": spe_values.tolist(),
            "is_outlier": is_outlier,
            "t2_limit": float(t2_lim),
            "spe_limit": float(spe_lim),
        }
        return clean(result)
    except (ValueError, TypeError, KeyError) as exc:
        return {"error": str(exc)}


_register("pca_predict")


# ---------------------------------------------------------------------------
# pls_predict
# ---------------------------------------------------------------------------


class PlsPredictInput(BaseModel):
    """Input contract for ``pls_predict``."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_params: dict[str, Any] = Field(
        ...,
        description="The model_params dict returned by fit_pls.",
    )
    new_data: list[list[float]] = Field(
        ...,
        min_length=1,
        description="New X data matrix as a list of rows (same number of columns as training X).",
    )


@tool_spec(
    name="pls_predict",
    description=(
        "Predict Y values for new observations using a fitted PLS model. "
        "Accepts the model_params dict returned by fit_pls and new X data rows. "
        "Returns predicted Y values (in original scale), scores, Hotelling's T-squared, "
        "and SPE diagnostics. "
        "Use this to make predictions on new data and check whether the new data is "
        "within the model's applicability domain."
    ),
    input_model=PlsPredictInput,
    examples="""
    # "Predict yield for new process conditions"
        -> ``pls_predict(model_params=<from fit_pls>, new_data=[[1,2,3],[4,5,6]])``
    """,
    category="multivariate",
)
def pls_predict(spec: PlsPredictInput) -> dict[str, Any]:
    """Predict Y from new X data using PLS model params."""
    try:
        _validate_matrix_shape(spec.new_data, "new_data")
        # SEC-25 (#274): cap every array-like sub-field of model_params
        # before any np.array(...) allocation.
        _validate_model_params(
            spec.model_params,
            keys_2d=("x_loadings", "y_loadings", "direct_weights", "beta_coefficients"),
            keys_1d=(
                "x_means", "x_stds", "y_means", "y_stds",
                "scaling_factor_for_scores", "spe_values",
            ),
        )
        from process_improve.multivariate.methods import hotellings_t2_limit, spe_calculation  # noqa: PLC0415

        x_means = np.array(spec.model_params["x_means"])
        x_stds = np.array(spec.model_params["x_stds"])
        y_means = np.array(spec.model_params["y_means"])
        y_stds = np.array(spec.model_params["y_stds"])
        direct_weights = np.array(spec.model_params["direct_weights"])
        x_loadings = np.array(spec.model_params["x_loadings"])
        y_loadings = np.array(spec.model_params["y_loadings"])
        scaling_factors = np.array(spec.model_params["scaling_factor_for_scores"])
        n_components = spec.model_params["n_components"]
        n_samples = spec.model_params["n_samples"]
        train_spe_values = np.array(spec.model_params["spe_values"])

        X_new = np.array(spec.new_data, dtype=float)
        X_scaled = (X_new - x_means) / x_stds

        scores = X_scaled @ direct_weights
        t2_values = np.sum((scores / scaling_factors) ** 2, axis=1)

        X_hat = scores @ x_loadings.T
        residuals = X_scaled - X_hat
        spe_values = np.sqrt(np.sum(residuals**2, axis=1))

        y_hat_scaled = scores @ y_loadings.T
        y_hat = y_hat_scaled * y_stds + y_means

        t2_lim = hotellings_t2_limit(
            conf_level=0.95,
            n_components=n_components,
            n_rows=n_samples,
        )
        spe_lim = spe_calculation(spe_values=train_spe_values, conf_level=0.95)

        result: dict[str, Any] = {
            "y_hat": y_hat.tolist(),
            "scores": scores.tolist(),
            "hotellings_t2": t2_values.tolist(),
            "spe": spe_values.tolist(),
            "t2_limit": float(t2_lim),
            "spe_limit": float(spe_lim),
        }
        return clean(result)
    except (ValueError, TypeError, KeyError) as exc:
        return {"error": str(exc)}


_register("pls_predict")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def get_multivariate_tool_specs() -> list[dict]:
    """Return tool specs for all multivariate tools registered in this module."""
    return get_tool_specs(names=_MULTIVARIATE_TOOL_NAMES)
