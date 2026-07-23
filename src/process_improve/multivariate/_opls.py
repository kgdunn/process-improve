# (c) Kevin Dunn, 2010-2026. MIT License.
"""Orthogonal Projections to Latent Structures (O-PLS) for a single response.

O-PLS (Trygg and Wold, 2002) is a variant of PLS that splits the systematic
variation in ``X`` into two parts: a single *predictive* component that is
correlated with the response ``y``, and one or more *Y-orthogonal* components
that carry systematic variation in ``X`` unrelated to ``y``. Filtering out the
orthogonal part leaves a model with the same predictive ability as an ordinary
PLS model fitted with the same total number of components, but with the
response-relevant variation concentrated in one component, which aids
interpretation.

This module implements the single-response case, O-PLS(1; ``n_orthogonal``),
following the notation of García-Carrión et al. (2025). That paper proves that
the *orthogonal space* isolated here is the same linear space as the *null
space* of an inverted PLS model with the same total number of components, so
:meth:`OPLS.invert` and :meth:`PLS.invert` describe the same set of designs;
O-PLS reaches it through a single division.

References
----------
J. Trygg and S. Wold, "Orthogonal projections to latent structures (O-PLS)",
Journal of Chemometrics, 16 (2002): 119-128, DOI: 10.1002/cem.695.

S. García-Carrión et al., "On the equivalence between null space and orthogonal
space in latent variable regression modeling", Journal of Chemometrics, 39
(2025): e70057, DOI: 10.1002/cem.70057.
"""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted, validate_data

from .._linalg import safe_inverse
from ._base import _LatentVariableModel, _LazyFrame
from ._common import epsqrt
from ._preprocessing import MCUVScaler

if typing.TYPE_CHECKING:
    from ._pls import DataMatrix


class OPLS(_LatentVariableModel, RegressorMixin, TransformerMixin, BaseEstimator):
    """Orthogonal PLS for a single response, O-PLS(1; ``n_orthogonal_components``).

    Extracts one Y-predictive component and ``n_orthogonal_components``
    Y-orthogonal components from ``X`` via the Trygg-Wold NIPALS algorithm. The
    predictive component carries the variation used to predict ``y``; the
    orthogonal components carry systematic ``X`` variation unrelated to ``y``.

    Parameters
    ----------
    n_orthogonal_components : int
        Number of Y-orthogonal components to extract (``Ao``). The model has a
        total of ``1 + n_orthogonal_components`` components, matching an ordinary
        PLS model with that many latent variables.
    scale : bool, default=True
        Mean-center and unit-variance-scale X and y internally before fitting
        (with :class:`MCUVScaler`). Predictions and ``beta_coefficients_`` are
        returned on the original data scale. Set ``False`` when the inputs are
        already scaled.
    max_iter : int, default=1000
        Reserved for API parity with :class:`PLS`; the closed-form single-
        response algorithm here does not iterate.
    tol : float, default=sqrt(machine epsilon)
        Numerical tolerance used when guarding rank-deficient projections.
    copy : bool, default=True
        Whether to copy X and y before fitting.

    Attributes (after fitting)
    --------------------------
    predictive_scores_ : pd.DataFrame of shape (n_samples, 1)
        Predictive score ``t_p``.
    predictive_weights_ : pd.Series of length n_features
        Predictive weight ``w_p``.
    predictive_loadings_ : pd.Series of length n_features
        Predictive loading ``p_p``.
    orthogonal_scores_ : pd.DataFrame of shape (n_samples, n_orthogonal)
        Y-orthogonal scores ``T_o``.
    orthogonal_weights_ : pd.DataFrame of shape (n_features, n_orthogonal)
        Y-orthogonal weights ``W_o``.
    orthogonal_loadings_ : pd.DataFrame of shape (n_features, n_orthogonal)
        Y-orthogonal loadings ``P_o``.
    y_loadings_ : float
        Predictive y-loading ``q_p`` (a scalar for a single response).
    beta_coefficients_ : pd.DataFrame of shape (n_features, 1)
        Regression coefficients linking X directly to y, on the original scale.
    scores_ : pd.DataFrame of shape (n_samples, 1 + n_orthogonal)
        Predictive score followed by the orthogonal scores, labelled
        ``["t_predictive", "t_orthogonal_1", ...]``. Lets the inherited
        :meth:`score_plot` draw the predictive-vs-orthogonal score plot.
    x_filtered_ : pd.DataFrame of shape (n_samples, n_features)
        The orthogonal-signal-corrected X (``X`` with the orthogonal variation
        removed), on the scaled fitting scale.
    spe_ : pd.DataFrame
        Per-row SPE after reconstructing X from all components.
    hotellings_t2_ : pd.DataFrame
        Cumulative Hotelling's T2 over the combined score space.

    See Also
    --------
    PLS : ordinary projection to latent structures; ``PLS.invert`` gives the
        same designs as :meth:`OPLS.invert` for a single response.

    References
    ----------
    J. Trygg and S. Wold, "Orthogonal projections to latent structures (O-PLS)",
    Journal of Chemometrics, 16 (2002): 119-128, DOI: 10.1002/cem.695.

    Examples
    --------
    >>> import pandas as pd
    >>> from process_improve.multivariate.methods import OPLS
    >>> X = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [1.0, 0, 1, 0]})
    >>> y = pd.DataFrame({"y": [1.0, 2, 3, 4]})
    >>> model = OPLS(n_orthogonal_components=1).fit(X, y)
    >>> model.predict(X).shape
    (4, 1)
    """

    _ATTRIBUTE_RENAMES: typing.ClassVar[dict[str, str]] = {}
    _RENAME_CONTEXT: typing.ClassVar[str] = "OPLS"

    # Lazily-built DataFrame views over the private ndarrays (ENG-18), matching
    # the PLS convention so the inherited score/SPE plots work unchanged.
    scores_ = _LazyFrame("_scores", index="_sample_index", columns="_component_names")
    spe_ = _LazyFrame("_spe", index="_sample_index", columns="_component_names")

    def __init__(
        self,
        n_orthogonal_components: int,
        *,
        scale: bool = True,
        max_iter: int = 1000,
        tol: float = epsqrt,
        copy: bool = True,
    ):
        self.n_orthogonal_components = n_orthogonal_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy

    def __sklearn_tags__(self):
        """Declare sklearn capability tags (sklearn 1.6+)."""
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.target_tags.single_output = True
        return tags

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # noqa: ANN001, ARG002
        """Return the component labels produced by :meth:`transform`."""
        check_is_fitted(self, "predictive_weights_")
        return np.asarray(self._component_names, dtype=object)

    def fit(self, X: DataMatrix, Y: DataMatrix) -> OPLS:  # noqa: C901, PLR0915
        """Fit the O-PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor block.
        Y : array-like of shape (n_samples,) or (n_samples, 1)
            Single response. A 2-D column or a 1-D vector are both accepted.

        Returns
        -------
        OPLS
            The fitted model (``self``).
        """
        if hasattr(Y, "ndim") and Y.ndim == 1:
            Y = Y.to_frame() if isinstance(Y, pd.Series) else pd.DataFrame(np.asarray(Y).reshape(-1, 1))

        sample_index: pd.Index | None = X.index if isinstance(X, pd.DataFrame) else None
        feature_columns: pd.Index | None = X.columns if isinstance(X, pd.DataFrame) else None
        X_arr = validate_data(
            self,
            X,
            reset=True,
            accept_sparse=False,
            ensure_min_samples=2,
            ensure_min_features=1,
            dtype="numeric",
            ensure_all_finite=True,
        )
        if feature_columns is None:
            feature_columns = pd.RangeIndex(X_arr.shape[1])
        if sample_index is None:
            sample_index = pd.RangeIndex(X_arr.shape[0])
        X = pd.DataFrame(X_arr, index=sample_index, columns=feature_columns)
        if not isinstance(Y, pd.DataFrame):
            Y = pd.DataFrame(Y, index=sample_index)
        if Y.shape[1] != 1:
            raise ValueError(f"OPLS supports a single response only; Y has {Y.shape[1]} columns.")

        self.n_samples_ = X.shape[0]
        self.n_targets_ = 1
        n_ortho = int(self.n_orthogonal_components)
        if n_ortho < 0:
            raise ValueError("n_orthogonal_components must be >= 0.")
        self.n_components = 1 + n_ortho

        self._x_scaler: MCUVScaler | None = None
        self._y_scaler: MCUVScaler | None = None
        if self.scale:
            self._x_scaler = MCUVScaler().fit(X)
            self._y_scaler = MCUVScaler().fit(Y)
            X = self._x_scaler.transform(X)
            Y = self._y_scaler.transform(Y)

        Xmat = X.to_numpy(dtype=float)
        y = Y.to_numpy(dtype=float)  # (N, 1)
        n_features = Xmat.shape[1]
        target_name = Y.columns

        # --- Trygg-Wold O-PLS NIPALS (single response) ---
        # Predictive weight direction (unit norm), fixed throughout.
        w_p = Xmat.T @ y
        w_p = w_p / np.linalg.norm(w_p)  # (K, 1)

        W_o = np.zeros((n_features, n_ortho))
        P_o = np.zeros((n_features, n_ortho))
        T_o = np.zeros((self.n_samples_, n_ortho))
        Xd = Xmat.copy()
        for i in range(n_ortho):
            t = Xd @ w_p  # (N, 1)
            p = (Xd.T @ t) / (t.T @ t).item()  # (K, 1)
            # Orthogonal weight: the part of the loading orthogonal to w_p.
            w_o = p - (w_p.T @ p).item() * w_p
            w_o = w_o / np.linalg.norm(w_o)
            t_o = Xd @ w_o
            p_o = (Xd.T @ t_o) / (t_o.T @ t_o).item()
            Xd = Xd - t_o @ p_o.T  # remove this orthogonal component
            W_o[:, [i]] = w_o
            P_o[:, [i]] = p_o
            T_o[:, [i]] = t_o

        # Predictive component on the filtered (orthogonal-corrected) X.
        t_p = Xd @ w_p  # (N, 1)
        p_p = (Xd.T @ t_p) / (t_p.T @ t_p).item()  # (K, 1)
        q_p = (y.T @ t_p).item() / (t_p.T @ t_p).item()  # scalar

        # OSC filter operator so predict/transform reproduce the fit exactly.
        if n_ortho > 0:
            filt = np.eye(n_features) - W_o @ safe_inverse(P_o.T @ W_o, what="(P_o' @ W_o)") @ P_o.T
        else:
            filt = np.eye(n_features)
        self._osc_filter = filt  # (K, K)
        self._w_p = w_p
        self._W_o = W_o
        self._P_o = P_o
        self._p_p = p_p
        self._q_p = q_p

        # beta on the (optionally) scaled space, then map back to original units.
        beta_scaled = (filt @ w_p) * q_p  # (K, 1)
        beta = beta_scaled.copy()
        if self._x_scaler is not None and self._y_scaler is not None:
            x_scale = self._x_scaler.scale_.to_numpy()[:, np.newaxis]
            y_scale = self._y_scaler.scale_.to_numpy()[np.newaxis, :]
            beta = beta * (y_scale / x_scale)

        # --- Public / backing attributes ---
        component_names = ["t_predictive"] + [f"t_orthogonal_{i}" for i in range(1, n_ortho + 1)]
        self._sample_index = X.index
        self._feature_names = X.columns
        self._component_names = component_names

        self.predictive_scores_ = pd.DataFrame(t_p, index=X.index, columns=["t_predictive"])
        self.predictive_weights_ = pd.Series(w_p.ravel(), index=X.columns, name="w_predictive")
        self.predictive_loadings_ = pd.Series(p_p.ravel(), index=X.columns, name="p_predictive")
        ortho_names = [f"o{i}" for i in range(1, n_ortho + 1)]
        self.orthogonal_scores_ = pd.DataFrame(T_o, index=X.index, columns=ortho_names)
        self.orthogonal_weights_ = pd.DataFrame(W_o, index=X.columns, columns=ortho_names)
        self.orthogonal_loadings_ = pd.DataFrame(P_o, index=X.columns, columns=ortho_names)
        self.y_loadings_ = q_p
        self.beta_coefficients_ = pd.DataFrame(beta, index=X.columns, columns=target_name)

        # Combined score matrix [t_p | T_o] backs scores_ and the geometry plots.
        self._scores = np.hstack([t_p, T_o])
        self.explained_variance_ = np.diag(self._scores.T @ self._scores) / max(1, self.n_samples_ - 1)
        self.scaling_factor_for_scores_ = pd.Series(
            np.sqrt(self.explained_variance_),
            index=component_names,
            name="Standard deviation per score",
        )

        # Filtered X and full reconstruction for SPE.
        self.x_filtered_ = pd.DataFrame(Xmat @ filt, index=X.index, columns=X.columns)
        x_hat = t_p @ p_p.T + T_o @ P_o.T
        residuals = Xmat - x_hat
        self._spe = np.sqrt(np.sum(residuals**2, axis=1)).reshape(-1, 1) * np.ones((1, self.n_components))

        # Cumulative Hotelling's T2 across the combined score space.
        std = self.scaling_factor_for_scores_.to_numpy()
        t2_cum = np.cumsum((self._scores / std) ** 2, axis=1)
        self.hotellings_t2_ = pd.DataFrame(t2_cum, index=X.index, columns=component_names)

        return self

    def transform(self, X: DataMatrix, Y: DataMatrix | None = None) -> pd.DataFrame:  # noqa: ARG002
        """Project new data and return the predictive score ``t_p``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to project.
        Y : ignored
            Present for sklearn pipeline API compatibility.

        Returns
        -------
        pd.DataFrame of shape (n_samples, 1)
            The predictive score for each row.
        """
        X_scaled, index = self._prepare(X)
        t_p = (X_scaled @ self._osc_filter) @ self._w_p
        return pd.DataFrame(t_p, index=index, columns=["t_predictive"])

    def correct(self, X: DataMatrix) -> pd.DataFrame:
        """Return the orthogonal-signal-corrected X (orthogonal variation removed).

        This is O-PLS's original use as a preprocessing filter: the returned
        matrix has the Y-orthogonal variation stripped out, on the scaled
        fitting scale.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        pd.DataFrame of shape (n_samples, n_features)
            The filtered X.
        """
        X_scaled, index = self._prepare(X)
        return pd.DataFrame(X_scaled @ self._osc_filter, index=index, columns=self._feature_names)

    def predict(self, X: DataMatrix) -> pd.DataFrame:
        """Predict the response for new observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        pd.DataFrame of shape (n_samples, 1)
            Predicted response on the original (un-scaled) y scale.
        """
        t_p = self.transform(X)
        y_hat = t_p.to_numpy() * self._q_p
        y_hat_df = pd.DataFrame(y_hat, index=t_p.index, columns=self.beta_coefficients_.columns)
        if self._y_scaler is not None:
            y_hat_df = self._y_scaler.inverse_transform(y_hat_df)
        return y_hat_df

    def invert(self, y_desired: float | np.ndarray | pd.Series | pd.DataFrame) -> Bunch:
        r"""Invert the O-PLS model: find inputs that yield a desired response.

        For a single response the inversion reduces to a single division. The
        predictive score that achieves the target is
        ``tau_p = y_desired / q_p``, and the *orthogonal space* (spanned by the
        orthogonal loadings ``P_o``) can be varied freely without changing the
        prediction. This is the same set of designs that :meth:`PLS.invert`
        returns as its null space (García-Carrión et al., 2025), reached here by
        one division instead of solving an underdetermined system.

        Parameters
        ----------
        y_desired : float, array-like, pandas Series/DataFrame
            Desired response on the original (un-scaled) y scale.

        Returns
        -------
        result : sklearn.utils.Bunch
            With keys:

            ``x_new`` : pd.Series of shape (n_features,)
                Input vector (minimum-norm; zero orthogonal contribution) on the
                original X scale.
            ``predictive_score`` : float
                The predictive score ``tau_p`` achieving the target.
            ``y_hat`` : float
                Prediction at ``x_new`` (equals ``y_desired`` up to rounding).
            ``orthogonal_space_basis`` : pd.DataFrame of shape (n_features, n_orthogonal)
                Basis of the orthogonal space in the input space (the orthogonal
                loadings). Adding any combination of these columns to ``x_new``
                leaves the prediction unchanged.
            ``orthogonal_space_dimension`` : int
                ``n_orthogonal_components``.

        See Also
        --------
        PLS.invert : the equivalent inversion via the null space.
        """
        check_is_fitted(self, "predictive_weights_")

        y_arr = np.atleast_1d(np.asarray(_as_scalar(y_desired), dtype=float))
        if y_arr.size != 1:
            raise ValueError("OPLS.invert expects a single desired response value.")
        y_value = float(y_arr[0])
        if not np.isfinite(y_value):
            raise ValueError("y_desired must be a finite value.")

        # Scale the target to the fitting space.
        if self._y_scaler is not None:
            y_scaled = (y_value - float(self._y_scaler.center_.iloc[0])) / float(self._y_scaler.scale_.iloc[0])
        else:
            y_scaled = y_value

        tau_p = y_scaled / self._q_p  # single division (paper Eq. 28)

        # Minimum-norm input: predictive contribution only, no orthogonal part.
        x_new_scaled = (tau_p * self._p_p).reshape(1, -1)  # (1, K)
        x_new_df = pd.DataFrame(x_new_scaled, columns=self._feature_names, index=[0])
        if self._x_scaler is not None:
            x_new_df = self._x_scaler.inverse_transform(x_new_df)

        y_hat_value = float(self.predict(x_new_df).to_numpy()[0, 0])

        return Bunch(
            x_new=pd.Series(x_new_df.to_numpy().ravel(), index=self._feature_names, name="x_new"),
            predictive_score=tau_p,
            y_hat=y_hat_value,
            orthogonal_space_basis=self.orthogonal_loadings_.copy(),
            orthogonal_space_dimension=int(self.n_orthogonal_components),
        )

    # ------------------------------------------------------------------ #
    def _prepare(self, X: DataMatrix) -> tuple[np.ndarray, pd.Index]:
        """Validate new X, capture its index, and move it to the scaled space."""
        check_is_fitted(self, "predictive_weights_")
        index: pd.Index | None = X.index if isinstance(X, pd.DataFrame) else None
        X_arr = validate_data(
            self, X, reset=False, accept_sparse=False, dtype="numeric", ensure_all_finite=True
        )
        if index is None:
            index = pd.RangeIndex(X_arr.shape[0])
        X_df = pd.DataFrame(X_arr, index=index, columns=self._feature_names)
        if self._x_scaler is not None:
            X_df = self._x_scaler.transform(X_df)
        return X_df.to_numpy(dtype=float), index


def _as_scalar(value: float | np.ndarray | pd.Series | pd.DataFrame) -> float | np.ndarray:
    """Extract the underlying scalar/array from assorted single-response inputs."""
    if isinstance(value, pd.DataFrame):
        return value.to_numpy().ravel()
    if isinstance(value, pd.Series):
        return value.to_numpy().ravel()
    return value
