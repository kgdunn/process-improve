import numpy as np
from scipy.stats import f, chi2, t
from sklearn.decomposition import PCA as PCA_sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# We will specialize this class later on.
PCA = PCA_sklearn


class PCA(PCA_sklearn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y=None) -> PCA:
        self = super().fit(X)

        self.N = X.shape[0]
        self.scaling_factor_for_scores = np.sqrt(self.explained_variance_)
        self.t_scores = super().fit_transform(X)
        self.hotellings_T2 = np.sum((self.t_scores / self.scaling_factor_for_scores) ** 2, axis=1)

        error_X = X - self.t_scores @ self.components_
        self.squared_prediction_error = np.sum(error_X ** 2, axis=1)
        return self

    def T2_limit(self, conf_level=0.95) -> float:
        """Returns the Hotelling's T2 value at the given level of confidence.

        Parameters
        ----------
        conf_level : float, optional
            Fractional confidence limit, less that 1.00; by default 0.95

        Returns
        -------
        float
            The Hotelling's T2 limit at the given level of confidence.
        """
        assert conf_level > 0.0
        assert conf_level < 1.0
        A, N = self.n_components, self.N
        return A * (N - 1) * (N + 1) / (N * (N - A)) * f.isf((1 - conf_level), A, N - A)

    def SPE_limit(self, conf_level=0.95) -> float:
        check_is_fitted(self, "squared_prediction_error")

        assert conf_level > 0.0
        assert conf_level < 1.0
        if self.N > 7:
            center_spe = self.squared_prediction_error.median()
            variance_spe = self.squared_prediction_error.var()  # TODO: robustify
        else:
            center_spe = self.squared_prediction_error.mean()
            variance_spe = self.squared_prediction_error.var()

        g = variance_spe / (2 * center_spe)
        h = (2 * center_spe ** 2) / variance_spe
        return chi2.ppf(conf_level, h) * g


# Create our own mean centering and scaling to unit variance (MCUV) class
# The default scaler in sklearn does not handle small datasets accurately, with ddof.
class MCUVScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.center_x_ = X.mean()
        self.scale_x_ = X.std(
            ddof=1
        )  # this is the key difference with "preprocessing.StandardScaler"
        self.scale_x_[self.scale_x_ == 0] = 1.0  # columns with no variance are left as-is.
        return self

    def transform(self, X):
        check_is_fitted(self, "center_x_")
        check_is_fitted(self, "scale_x_")

        X = X.copy()
        return (X - self.center_x_) / self.scale_x_

    def inverse_transform(self, X):
        check_is_fitted(self, "center_x_")
        check_is_fitted(self, "scale_x_")

        X = X.copy()
        return X * self.scale_x_ + self.center_x_

