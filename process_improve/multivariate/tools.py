"""(c) Kevin Dunn, 2010-2025. MIT License.

Agent-callable tool wrappers for multivariate analysis (PCA, PLS, scaling).

Each function in this module is decorated with ``@tool_spec`` so it can be
passed directly to an LLM tool-use API (e.g. Anthropic ``tools=``).
The wrappers accept plain JSON-serialisable inputs (lists of numbers, strings,
booleans) and always return JSON-serialisable ``dict`` results.

Import all specs at once::

    from process_improve.multivariate.tools import get_multivariate_tool_specs
    # or get everything registered so far
    from process_improve.tool_spec import get_tool_specs

Dispatch a tool call returned by the model::

    from process_improve.tool_spec import execute_tool_call
    result = execute_tool_call(block.name, block.input)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from process_improve.tool_spec import clean, get_tool_specs, tool_spec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MULTIVARIATE_TOOL_NAMES: list[str] = []


def _register(name: str) -> None:
    _MULTIVARIATE_TOOL_NAMES.append(name)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


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
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Data matrix as a list of rows, where each row is a list of numeric values.",
                    "minItems": 3,
                },
                "n_components": {
                    "type": "integer",
                    "description": "Number of principal components to extract.",
                    "minimum": 1,
                },
                "column_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional names for each column/variable. Length must match the number of columns.",
                },
            },
            "required": ["data", "n_components"],
        }
    },
    examples="""
    # "Run PCA with 2 components on my data"
        -> ``fit_pca(data=[[1,2,3],[4,5,6],[7,8,9]], n_components=2)``

    # "Fit a 3-component PCA with named variables"
        -> ``fit_pca(data=[[...],[...]], n_components=3, column_names=["temp","pressure","flow"])``
    """,
    category="multivariate",
)
def fit_pca(
    *,
    data: list[list[float]],
    n_components: int,
    column_names: list[str] | None = None,
) -> dict[str, Any]:
    """Fit a PCA model to the given data; see tool spec for details."""
    try:
        from process_improve.multivariate.methods import PCA, MCUVScaler

        df = pd.DataFrame(data, columns=column_names)
        scaler = MCUVScaler().fit(df)
        X_scaled = scaler.transform(df)

        model = PCA(n_components=n_components).fit(X_scaled)

        # Detect outliers
        outliers = model.detect_outliers(conf_level=0.95)

        # Build serialisable model_params for pca_predict
        model_params = {
            "loadings": model.loadings_.values.tolist(),
            "means": scaler.center_.values.tolist(),
            "stds": scaler.scale_.values.tolist(),
            "n_components": n_components,
            "scaling_factor_for_scores": model.scaling_factor_for_scores_.values.tolist(),
            "n_samples": int(model.n_samples_),
            "spe_values": model.spe_.iloc[:, -1].values.tolist(),
        }

        result: dict[str, Any] = {
            "n_components": n_components,
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
    except Exception as exc:
        return {"error": str(exc)}


_register("fit_pca")


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
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "x_data": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "X data matrix as a list of rows.",
                    "minItems": 3,
                },
                "y_data": {
                    "type": "array",
                    "description": (
                        "Y data as a list of lists (multiple responses) or a flat list of numbers "
                        "(single response)."
                    ),
                    "minItems": 3,
                },
                "n_components": {
                    "type": "integer",
                    "description": "Number of latent components to extract.",
                    "minimum": 1,
                },
                "x_column_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional names for X columns.",
                },
                "y_column_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional names for Y columns.",
                },
            },
            "required": ["x_data", "y_data", "n_components"],
        }
    },
    examples="""
    # "Build a PLS model with 2 components predicting yield from process variables"
        -> ``fit_pls(x_data=[[...],[...]], y_data=[10.1, 10.5, ...], n_components=2)``

    # "Fit PLS with named X and Y variables"
        -> ``fit_pls(x_data=[[...]], y_data=[[...]], n_components=3,
                     x_column_names=["temp","flow"], y_column_names=["yield"])``
    """,
    category="multivariate",
)
def fit_pls(
    *,
    x_data: list[list[float]],
    y_data: list[list[float]] | list[float],
    n_components: int,
    x_column_names: list[str] | None = None,
    y_column_names: list[str] | None = None,
) -> dict[str, Any]:
    """Fit a PLS model to X and Y data; see tool spec for details."""
    try:
        from process_improve.multivariate.methods import PLS, MCUVScaler

        X = pd.DataFrame(x_data, columns=x_column_names)

        # Handle y_data: flat list (single response) or list of lists (multiple)
        if y_data and not isinstance(y_data[0], list):
            Y = pd.DataFrame(y_data, columns=y_column_names or ["y"])
        else:
            Y = pd.DataFrame(y_data, columns=y_column_names)

        scaler_x = MCUVScaler().fit(X)
        scaler_y = MCUVScaler().fit(Y)
        X_scaled = scaler_x.transform(X)
        Y_scaled = scaler_y.transform(Y)

        model = PLS(n_components=n_components, scale=False).fit(X_scaled, Y_scaled)

        # Build serialisable model_params for pls_predict
        model_params = {
            "x_loadings": model.x_loadings_.values.tolist(),
            "y_loadings": model.y_loadings_.values.tolist(),
            "direct_weights": model.direct_weights_.values.tolist(),
            "beta_coefficients": model.beta_coefficients_.values.tolist(),
            "x_means": scaler_x.center_.values.tolist(),
            "x_stds": scaler_x.scale_.values.tolist(),
            "y_means": scaler_y.center_.values.tolist(),
            "y_stds": scaler_y.scale_.values.tolist(),
            "n_components": n_components,
            "scaling_factor_for_scores": model.scaling_factor_for_scores_.values.tolist(),
            "n_samples": int(model.n_samples_),
            "spe_values": model.spe_.iloc[:, -1].values.tolist(),
        }

        result: dict[str, Any] = {
            "n_components": n_components,
            "r2x_cumulative": model.r2_cumulative_.values.tolist(),
            "r2_per_component": model.r2_per_component_.values.tolist(),
            "rmse": model.rmse_.values.tolist(),
            "coefficients": model.beta_coefficients_.values.tolist(),
            "model_params": model_params,
        }
        return clean(result)
    except Exception as exc:
        return {"error": str(exc)}


_register("fit_pls")


@tool_spec(
    name="scale_data",
    description=(
        "Mean-centre and scale a data matrix to unit variance (MCUV scaling). "
        "Returns the scaled data along with the column means and standard deviations used. "
        "This is the standard preprocessing step before PCA or PLS. "
        "Use this when you need the scaled data for further analysis or want to inspect "
        "the scaling parameters."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Data matrix as a list of rows.",
                    "minItems": 1,
                },
                "column_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional names for each column/variable.",
                },
            },
            "required": ["data"],
        }
    },
    examples="""
    # "Scale my data to zero mean and unit variance"
        -> ``scale_data(data=[[1,2],[3,4],[5,6]])``

    # "Scale with named columns"
        -> ``scale_data(data=[[10,20],[30,40]], column_names=["temp","pressure"])``
    """,
    category="multivariate",
)
def scale_data(
    *,
    data: list[list[float]],
    column_names: list[str] | None = None,
) -> dict[str, Any]:
    """Mean-centre and scale data to unit variance; see tool spec for details."""
    try:
        from process_improve.multivariate.methods import MCUVScaler

        df = pd.DataFrame(data, columns=column_names)
        scaler = MCUVScaler().fit(df)
        scaled = scaler.transform(df)

        result: dict[str, Any] = {
            "scaled_data": scaled.values.tolist(),
            "means": scaler.center_.values.tolist(),
            "stds": scaler.scale_.values.tolist(),
        }
        return clean(result)
    except Exception as exc:
        return {"error": str(exc)}


_register("scale_data")


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
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Data matrix as a list of rows.",
                    "minItems": 3,
                },
                "n_components": {
                    "type": "integer",
                    "description": "Number of PCA components to use for outlier detection.",
                    "minimum": 1,
                },
                "conf_level": {
                    "type": "number",
                    "description": "Confidence level for the statistical limits (default 0.95).",
                    "minimum": 0.8,
                    "maximum": 0.999,
                },
            },
            "required": ["data", "n_components"],
        }
    },
    examples="""
    # "Are there multivariate outliers in my data?"
        -> ``detect_multivariate_outliers(data=[[...],[...]], n_components=2)``

    # "Detect outliers at 99% confidence"
        -> ``detect_multivariate_outliers(data=[[...]], n_components=2, conf_level=0.99)``
    """,
    category="multivariate",
)
def detect_multivariate_outliers(
    *,
    data: list[list[float]],
    n_components: int,
    conf_level: float = 0.95,
) -> dict[str, Any]:
    """Detect multivariate outliers via PCA diagnostics; see tool spec for details."""
    try:
        from process_improve.multivariate.methods import PCA, MCUVScaler

        df = pd.DataFrame(data)
        scaler = MCUVScaler().fit(df)
        X_scaled = scaler.transform(df)

        model = PCA(n_components=n_components).fit(X_scaled)
        outliers = model.detect_outliers(conf_level=conf_level)

        t2_lim = float(model.hotellings_t2_limit(conf_level=conf_level))
        spe_lim = float(model.spe_limit(conf_level=conf_level))

        result: dict[str, Any] = {
            "outlier_indices": [o["observation"] for o in outliers],
            "outlier_details": outliers,
            "t2_limit": t2_lim,
            "spe_limit": spe_lim,
        }
        return clean(result)
    except Exception as exc:
        return {"error": str(exc)}


_register("detect_multivariate_outliers")


@tool_spec(
    name="pca_predict",
    description=(
        "Project new observations into an existing PCA model and compute diagnostics. "
        "Accepts the model_params dict returned by fit_pca and new data rows. "
        "Returns scores, Hotelling's T-squared, SPE, and whether each observation is an "
        "outlier (T-squared exceeding the training-data-based limit). "
        "Use this to monitor new data against a PCA model built on historical data."
    ),
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "model_params": {
                    "type": "object",
                    "description": "The model_params dict returned by fit_pca.",
                },
                "new_data": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "New data matrix as a list of rows (same number of columns as training data).",
                    "minItems": 1,
                },
            },
            "required": ["model_params", "new_data"],
        }
    },
    examples="""
    # "Score new observations against my PCA model"
        -> ``pca_predict(model_params=<from fit_pca>, new_data=[[1,2,3],[4,5,6]])``
    """,
    category="multivariate",
)
def pca_predict(
    *,
    model_params: dict[str, Any],
    new_data: list[list[float]],
) -> dict[str, Any]:
    """Project new data into a PCA model; see tool spec for details."""
    try:
        from process_improve.multivariate.methods import hotellings_t2_limit, spe_calculation

        means = np.array(model_params["means"])
        stds = np.array(model_params["stds"])
        loadings = np.array(model_params["loadings"])
        scaling_factors = np.array(model_params["scaling_factor_for_scores"])
        n_components = model_params["n_components"]
        n_samples = model_params["n_samples"]
        train_spe_values = np.array(model_params["spe_values"])

        X_new = np.array(new_data, dtype=float)
        X_scaled = (X_new - means) / stds

        # Scores = X_scaled @ loadings
        scores = X_scaled @ loadings

        # Hotelling's T2 (cumulative across all components)
        t2_values = np.sum((scores / scaling_factors) ** 2, axis=1)

        # SPE: residual after reconstruction
        X_hat = scores @ loadings.T
        residuals = X_scaled - X_hat
        spe_values = np.sqrt(np.sum(residuals**2, axis=1))

        # Compute limits for outlier flagging
        t2_lim = hotellings_t2_limit(
            conf_level=0.95,
            n_components=n_components,
            n_rows=n_samples,
        )
        spe_lim = spe_calculation(spe_values=train_spe_values, conf_level=0.95)

        is_outlier = [(bool(t2 > t2_lim) or bool(spe > spe_lim)) for t2, spe in zip(t2_values, spe_values)]

        result: dict[str, Any] = {
            "scores": scores.tolist(),
            "hotellings_t2": t2_values.tolist(),
            "spe": spe_values.tolist(),
            "is_outlier": is_outlier,
            "t2_limit": float(t2_lim),
            "spe_limit": float(spe_lim),
        }
        return clean(result)
    except Exception as exc:
        return {"error": str(exc)}


_register("pca_predict")


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
    input_schema={
        "json": {
            "type": "object",
            "properties": {
                "model_params": {
                    "type": "object",
                    "description": "The model_params dict returned by fit_pls.",
                },
                "new_data": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "New X data matrix as a list of rows (same number of columns as training X).",
                    "minItems": 1,
                },
            },
            "required": ["model_params", "new_data"],
        }
    },
    examples="""
    # "Predict yield for new process conditions"
        -> ``pls_predict(model_params=<from fit_pls>, new_data=[[1,2,3],[4,5,6]])``
    """,
    category="multivariate",
)
def pls_predict(
    *,
    model_params: dict[str, Any],
    new_data: list[list[float]],
) -> dict[str, Any]:
    """Predict Y from new X data using PLS model params; see tool spec for details."""
    try:
        from process_improve.multivariate.methods import hotellings_t2_limit, spe_calculation

        x_means = np.array(model_params["x_means"])
        x_stds = np.array(model_params["x_stds"])
        y_means = np.array(model_params["y_means"])
        y_stds = np.array(model_params["y_stds"])
        direct_weights = np.array(model_params["direct_weights"])
        x_loadings = np.array(model_params["x_loadings"])
        y_loadings = np.array(model_params["y_loadings"])
        scaling_factors = np.array(model_params["scaling_factor_for_scores"])
        n_components = model_params["n_components"]
        n_samples = model_params["n_samples"]
        train_spe_values = np.array(model_params["spe_values"])

        X_new = np.array(new_data, dtype=float)
        X_scaled = (X_new - x_means) / x_stds

        # Scores via direct weights: T = X_scaled @ W*
        scores = X_scaled @ direct_weights

        # Hotelling's T2
        t2_values = np.sum((scores / scaling_factors) ** 2, axis=1)

        # SPE: residual after X reconstruction
        X_hat = scores @ x_loadings.T
        residuals = X_scaled - X_hat
        spe_values = np.sqrt(np.sum(residuals**2, axis=1))

        # Y prediction (in scaled space, then back to original)
        y_hat_scaled = scores @ y_loadings.T
        y_hat = y_hat_scaled * y_stds + y_means

        # Compute limits
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
    except Exception as exc:
        return {"error": str(exc)}


_register("pls_predict")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def get_multivariate_tool_specs() -> list[dict]:
    """Return tool specs for all multivariate tools registered in this module."""
    return get_tool_specs(names=_MULTIVARIATE_TOOL_NAMES)
