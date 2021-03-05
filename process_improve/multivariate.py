import numpy as np
from scipy.stats import f, chi2
from sklearn.decomposition import PCA as PCA_sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .robust import Sn


class PCA(PCA_sklearn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y=None) -> PCA_sklearn:
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

    def SPE_limit(self, conf_level=0.95, robust=True) -> float:
        check_is_fitted(self, "squared_prediction_error")

        assert conf_level > 0.0
        assert conf_level < 1.0
        if (self.N > 15) and robust:
            # The "15" is just a rough cut off, above which the robust estimators would
            # start to work well. Below which we can get doubtful results.
            center_spe = self.squared_prediction_error.median()
            variance_spe = Sn(self.squared_prediction_error) ** 2
        else:
            center_spe = self.squared_prediction_error.mean()
            variance_spe = self.squared_prediction_error.var()

        g = variance_spe / (2 * center_spe)
        h = (2 * center_spe ** 2) / variance_spe
        return chi2.ppf(conf_level, h) * g

    def ellipse_coordinates(
        self,
        score_horiz: int,
        score_vert: int,
        T2_limit_conf_level: float = 0.05,
        n_points: int = 100,
    ) -> tuple:
        """Get the (score_horiz, score_vert) coordinate pairs that form the T2 ellipse when
            plotting the score `score_horiz` on the horizontal axis and `score_vert` on the
            vertical axis.

            Scores are referred to by number, starting at 1 and ending with `model.components_`


        Parameters
        ----------
        score_horiz : int
            [description]
        score_vert : int
            [description]
        T2_limit_conf_level : float
            The `conf_level` confidence value: e.g. 0.95 is for the 95% confidence limit.
        n_points : int, optional
            Number of points to use in the ellipse; by default 100.

        Returns
        -------
        tuple of 2 elements; the first for the x-axis; the second for the y-axis.
            Returns `n_points` equispaced points that can be used to plot an ellipse.

        Background
        ----------

        Equation of ellipse in *canonical* form (http://en.wikipedia.org/wiki/Ellipse)

            (t_horiz/s_h)^2 + (t_vert/s_v)^2  =  T2_limit_alpha
            s_horiz = stddev(T_horiz)
            s_vert  = stddev(T_vert)
            T2_limit_alpha = T2 confidence limit at a given alpha value

        Equation of ellipse, *parametric* form (http://en.wikipedia.org/wiki/Ellipse):

            t_horiz = sqrt(T2_limit_alpha)*s_h*cos(t)
            t_vert  = sqrt(T2_limit_alpha)*s_v*sin(t)

            where t ranges between 0 and 2*pi.
        """
        assert score_horiz >= 1
        assert score_vert >= 1
        assert score_horiz <= self.n_components
        assert score_vert <= self.n_components
        assert T2_limit_conf_level > 0
        assert T2_limit_conf_level < 1
        s_h = self.scaling_factor_for_scores[score_horiz - 1]
        s_v = self.scaling_factor_for_scores[score_vert - 1]
        T2_limit = self.T2_limit(T2_limit_conf_level)
        dt = 2 * np.pi / (n_points - 1)
        steps = np.linspace(0, n_points - 1, n_points)
        x = np.cos(steps * dt) * np.sqrt(T2_limit) * s_h
        y = np.sin(steps * dt) * np.sqrt(T2_limit) * s_v
        return x, y


class MCUVScaler(BaseEstimator, TransformerMixin):
    """
    Create our own mean centering and scaling to unit variance (MCUV) class
    The default scaler in sklearn does not handle small datasets accurately, with ddof.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.center_x_ = X.mean()
        # this is the key difference with "preprocessing.StandardScaler"
        self.scale_x_ = X.std(ddof=1)
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
