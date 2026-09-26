# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Latent-variable methods for categorical and count data (#176, #177).

Every other method in :mod:`process_improve.multivariate` assumes a continuous
numeric matrix. Count data is everywhere in quality work nevertheless: defect type
against production line, failure mode against asset, out-of-specification reason
against supplier. The table is small, but reading it by eye does not scale, and a
PCA of it is wrong, because a PCA weights each cell by its size rather than by how
far it departs from what independence would predict.

Correspondence analysis is the method built for that table. It measures each row's
profile against the average profile by the chi-squared distance, which divides by the
expected frequency, so a rare category that departs strongly from independence counts
for as much as a common one that departs a little. The decomposition of that
chi-squared structure then gives a map on which rows with similar profiles sit
together, and a row near a column means that pair co-occurs more often than
independence would predict: a multivariate Pareto chart.

References
----------
M. Greenacre, "Correspondence Analysis in Practice", 3rd edition, Chapman and
Hall/CRC, 2017. The staff-by-smoking table used in the tests is his.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._common import DataMatrix, SpecificationWarning

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - exercised only without the plotting extra
    from process_improve._extras import _MissingExtra

    go = _MissingExtra("plotly", "plotting")  # type: ignore[assignment]

from process_improve.visualization.themes import DEFAULT_THEME


def _as_table(X: DataMatrix, what: str) -> pd.DataFrame:
    """Coerce a contingency table to a labelled float DataFrame and check it is one.

    Raises
    ------
    ValueError
        If the table is not 2-D, has fewer than two rows or columns, contains a
        missing, infinite or negative cell, or has a row or column that sums to zero.
    """
    table = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    if table.ndim != 2 or min(table.shape) < 2:
        raise ValueError(f"{what} must be a 2-D table with at least two rows and two columns; got {table.shape}.")
    values = table.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{what} must contain only finite counts (no NaN or infinity).")
    if np.any(values < 0):
        raise ValueError(f"{what} must not contain negative counts.")
    for axis, label, names in ((1, "Rows", table.index), (0, "Columns", table.columns)):
        empty = values.sum(axis=axis) == 0
        if np.any(empty):
            raise ValueError(
                f"{label} {list(names[empty])} of {what} sum to zero. A category that never occurs has no "
                "profile to compare, and its mass of zero would divide the chi-squared distance by zero. "
                "Drop it before fitting."
            )
    return table.astype(float)


#: Principal inertia below which an axis is numerical noise rather than structure.
#: Inertia is chi-squared over n, dimensionless and at most ``min(I, J) - 1``, so an
#: axis this small carries nothing on any real table: an exactly independent table
#: comes out around 1e-32, and would otherwise report its noise as "82% on axis 1".
_NULL_INERTIA = 1e-12


def _flip_signs(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fix the arbitrary sign of each singular pair, so a refit gives the same map.

    Each axis is flipped so that its largest-magnitude left singular vector entry is
    positive, the same convention the package's PCA uses for its loadings.
    """
    for k in range(left.shape[1]):
        if left[np.argmax(np.abs(left[:, k])), k] < 0:
            left[:, k] *= -1.0
            right[:, k] *= -1.0
    return left, right


class CA(TransformerMixin, BaseEstimator):
    r"""Correspondence analysis of a two-way contingency table.

    With :math:`P = N / n` the table of relative frequencies, :math:`r` and :math:`c`
    its row and column masses (the marginal proportions), the matrix of standardised
    residuals from independence,

    .. math::

        S = D_r^{-1/2} \, (P - r c^T) \, D_c^{-1/2},

    is decomposed as :math:`S = U \Sigma V^T`. The squared singular values are the
    principal inertias; their sum is the total inertia, :math:`\chi^2 / n`. The
    principal coordinates are :math:`F = D_r^{-1/2} U \Sigma` for the rows and
    :math:`G = D_c^{-1/2} V \Sigma` for the columns, and Euclidean distance between
    two rows on the full map is exactly the chi-squared distance between their
    profiles.

    Parameters
    ----------
    n_components : int, optional
        Number of axes to keep, default 2. A table with :math:`I` rows and :math:`J`
        columns has at most :math:`\min(I, J) - 1` axes; a larger request is clamped,
        with a :class:`~process_improve.multivariate._common.SpecificationWarning`.

    Attributes
    ----------
    n_components_ : int
        Axes actually kept, after clamping.
    eigenvalues_ : np.ndarray of shape (n_components,)
        Principal inertia of each kept axis.
    total_inertia_ : float
        Total inertia of the table, over every axis, kept or not: :math:`\chi^2 / n`.
    explained_inertia_ : np.ndarray of shape (n_components,)
        Share of the total inertia on each kept axis.
    row_masses_, column_masses_ : pd.Series
        Marginal proportions of the table.
    row_coordinates_, column_coordinates_ : pd.DataFrame
        Principal coordinates, one row per category, one column per axis.
    row_contributions_, column_contributions_ : pd.DataFrame
        Share of each axis's inertia contributed by each category; each column sums
        to 1. A category that contributes little to an axis is not what defines it,
        however far out it plots.
    row_cos2_, column_cos2_ : pd.DataFrame
        Squared cosine of each category with each axis: the share of that category's
        own inertia the axis represents. A category that plots near the origin with
        a low cos2 is not "average", it is poorly represented on these axes.

    References
    ----------
    M. Greenacre, "Correspondence Analysis in Practice", 3rd edition, 2017.

    Examples
    --------
    >>> ca = CA(n_components=2).fit(defects_by_line)      # doctest: +SKIP
    >>> ca.explained_inertia_                             # doctest: +SKIP
    >>> ca.map_plot()                                     # doctest: +SKIP
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X: DataMatrix, y: object = None) -> CA:  # noqa: ARG002
        """Fit the correspondence analysis to a contingency table.

        Parameters
        ----------
        X : array-like of shape (n_rows, n_columns)
            The table of non-negative counts. Row and column labels are kept when it
            is a DataFrame.
        y : ignored
            Accepted for :class:`~sklearn.pipeline.Pipeline` compatibility.

        Returns
        -------
        CA
            ``self``, fitted.

        Raises
        ------
        ValueError
            If ``n_components`` is not positive, or the table is not a valid
            contingency table (see :func:`_as_table`).
        """
        if int(self.n_components) < 1:
            raise ValueError(f"n_components must be at least 1; got {self.n_components}.")
        table = _as_table(X, "X")
        proportions = table.to_numpy() / table.to_numpy().sum()
        row_mass = proportions.sum(axis=1)
        column_mass = proportions.sum(axis=0)

        residuals = (proportions - np.outer(row_mass, column_mass)) / np.sqrt(np.outer(row_mass, column_mass))
        left, singular, right_t = np.linalg.svd(residuals, full_matrices=False)
        # At most min(I, J) - 1 axes, since centring on the average profile removes
        # one; fewer when the table is rank-deficient. An axis with no inertia is not
        # kept, because every relative quantity on it (its share, the contributions to
        # it) would be noise divided by noise.
        rank = int(np.sum(singular[: min(table.shape) - 1] ** 2 > _NULL_INERTIA))
        if rank == 0:
            raise ValueError(
                "The table shows no association between its rows and columns: every row has the same "
                "profile, so the total inertia is zero and there are no axes to find. That is a result, "
                "not a fault in the data, but there is no map to draw from it."
            )
        left, right = _flip_signs(left[:, :rank], right_t.T[:, :rank])
        singular = singular[:rank]

        keep = int(self.n_components)
        if keep > rank:
            warnings.warn(
                f"Asked for {keep} axes, but this {table.shape[0]} x {table.shape[1]} table has {rank} with "
                f"any inertia; keeping {rank}.",
                SpecificationWarning,
                stacklevel=2,
            )
            keep = rank

        all_rows = left / np.sqrt(row_mass)[:, np.newaxis] * singular
        all_columns = right / np.sqrt(column_mass)[:, np.newaxis] * singular
        eigenvalues = singular**2

        self.n_components_ = keep
        self.eigenvalues_ = eigenvalues[:keep]
        # Every axis with inertia, kept or not: a correction that sums over all of them
        # must not change with how many the caller chose to look at.
        self._all_eigenvalues = eigenvalues
        self.total_inertia_ = float(eigenvalues.sum())
        self.explained_inertia_ = self.eigenvalues_ / self.total_inertia_
        self.row_masses_ = pd.Series(row_mass, index=table.index, name="mass")
        self.column_masses_ = pd.Series(column_mass, index=table.columns, name="mass")

        axes = list(range(1, keep + 1))
        self.row_coordinates_ = pd.DataFrame(all_rows[:, :keep], index=table.index, columns=axes)
        self.column_coordinates_ = pd.DataFrame(all_columns[:, :keep], index=table.columns, columns=axes)
        self.row_contributions_ = pd.DataFrame(
            row_mass[:, np.newaxis] * all_rows[:, :keep] ** 2 / self.eigenvalues_, index=table.index, columns=axes
        )
        self.column_contributions_ = pd.DataFrame(
            column_mass[:, np.newaxis] * all_columns[:, :keep] ** 2 / self.eigenvalues_,
            index=table.columns,
            columns=axes,
        )
        # Squared distance to the centroid over *every* axis: a category's whole inertia.
        self.row_cos2_ = pd.DataFrame(
            all_rows[:, :keep] ** 2 / (all_rows**2).sum(axis=1, keepdims=True), index=table.index, columns=axes
        )
        self.column_cos2_ = pd.DataFrame(
            all_columns[:, :keep] ** 2 / (all_columns**2).sum(axis=1, keepdims=True),
            index=table.columns,
            columns=axes,
        )
        # Standard coordinates project supplementary profiles (the transition formula).
        self._row_standard = left[:, :keep] / np.sqrt(row_mass)[:, np.newaxis]
        self._column_standard = right[:, :keep] / np.sqrt(column_mass)[:, np.newaxis]
        self._columns = table.columns
        self._rows = table.index
        return self

    def transform(self, X: DataMatrix) -> pd.DataFrame:
        """Place rows on the fitted map, from their profiles.

        A supplementary row is positioned by the transition formula: its profile (the
        row divided by its total) times the column standard coordinates. It takes no
        part in defining the axes, which is what lets a new batch, or a row held out
        of the fit on purpose, be shown against the existing structure. Transforming
        the fitted table itself returns :attr:`row_coordinates_`.

        Parameters
        ----------
        X : array-like of shape (n_rows, n_columns)
            Counts with the same columns, in the same order, as the fitted table.

        Returns
        -------
        pd.DataFrame
            Principal coordinates, one row per input row.

        Raises
        ------
        ValueError
            If the column count differs from the fitted table's, or the rows are not
            valid counts.
        """
        check_is_fitted(self, "row_coordinates_")
        rows = self._supplementary(X, expected=len(self._columns), axis=1, what="rows")
        profiles = rows.to_numpy() / rows.to_numpy().sum(axis=1, keepdims=True)
        return pd.DataFrame(profiles @ self._column_standard, index=rows.index, columns=self.row_coordinates_.columns)

    def transform_columns(self, X: DataMatrix) -> pd.DataFrame:
        """Place supplementary *columns* on the fitted map.

        The column counterpart of :meth:`transform`: ``X`` has one row per fitted row
        category and one column per supplementary column, such as an outcome label
        (pass / fail) cross-tabulated against the fitted rows.

        Parameters
        ----------
        X : array-like of shape (n_rows, n_new_columns)
            Counts with the same rows, in the same order, as the fitted table.

        Returns
        -------
        pd.DataFrame
            Principal coordinates, one row per supplementary column.

        Raises
        ------
        ValueError
            If the row count differs from the fitted table's, or the columns are not
            valid counts.
        """
        check_is_fitted(self, "column_coordinates_")
        columns = self._supplementary(X, expected=len(self._rows), axis=0, what="columns")
        profiles = columns.to_numpy() / columns.to_numpy().sum(axis=0, keepdims=True)
        return pd.DataFrame(
            profiles.T @ self._row_standard, index=columns.columns, columns=self.column_coordinates_.columns
        )

    @staticmethod
    def _supplementary(X: DataMatrix, *, expected: int, axis: int, what: str) -> pd.DataFrame:
        """Validate supplementary counts; ``axis`` is the dimension that must match the fit."""
        table = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        if table.ndim != 2 or table.shape[axis] != expected:
            raise ValueError(
                f"Supplementary {what} must match the fitted table along the shared dimension: "
                f"expected {expected}, got shape {table.shape}."
            )
        values = table.to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"Supplementary {what} must be finite, non-negative counts.")
        if np.any(values.sum(axis=1 - axis) == 0):
            raise ValueError(f"A supplementary {what[:-1]} sums to zero, so it has no profile to place.")
        return table.astype(float)

    def map_plot(self, axis_x: int = 1, axis_y: int = 2, settings: dict | None = None) -> go.Figure:
        """Draw the symmetric map: rows and columns together, both in principal coordinates.

        Distances *within* the rows, and within the columns, are chi-squared
        distances. A distance between a row point and a column point has no such
        meaning on the symmetric map; what does is the direction: a row lying out
        toward a column co-occurs with it more than independence would predict.

        Parameters
        ----------
        axis_x, axis_y : int, optional
            The axes to plot, counted from 1. Defaults 1 and 2.
        settings : dict, optional
            Default settings::

                {
                    "title": "Correspondence analysis map",  # str: plot title
                    "row_color": None,        # str|None: marker colour for rows
                    "column_color": None,     # str|None: marker colour for columns
                    "html_image_height": 600, # int: image height in pixels
                    "html_aspect_ratio_w_over_h": 1.0,  # float: width over height
                    "template": "pi_journal", # str: registered Plotly theme name
                }

        Returns
        -------
        go.Figure

        Raises
        ------
        ValueError
            If an axis is outside ``1 .. n_components_``.
        """
        check_is_fitted(self, "row_coordinates_")

        class Settings(BaseModel):
            """Validated display settings for the correspondence analysis map."""

            title: str = "Correspondence analysis map"
            row_color: str | None = None
            column_color: str | None = None
            html_image_height: float = 600.0
            html_aspect_ratio_w_over_h: float = 1.0
            template: str = DEFAULT_THEME

        setdict = Settings(**(settings or {})).model_dump()
        for axis in (axis_x, axis_y):
            if not 1 <= axis <= self.n_components_:
                raise ValueError(f"Axes run from 1 to {self.n_components_}; got {axis}.")

        fig = go.Figure()
        for frame, name, colour, symbol in (
            (self.row_coordinates_, "rows", setdict["row_color"], "circle"),
            (self.column_coordinates_, "columns", setdict["column_color"], "triangle-up"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=frame[axis_x],
                    y=frame[axis_y],
                    mode="markers+text",
                    text=[str(label) for label in frame.index],
                    textposition="top center",
                    name=name,
                    marker={"color": colour, "symbol": symbol, "size": 10},
                    hovertemplate="%{text}: (%{x:.3f}, %{y:.3f})<extra>" + name + "</extra>",
                )
            )
        share = 100 * self.explained_inertia_
        fig.update_layout(
            template=setdict["template"],
            title_text=setdict["title"],
            xaxis={"title_text": f"Axis {axis_x} ({share[axis_x - 1]:.1f}% of inertia)", "zeroline": True},
            # Equal scaling: distances on this map are the point of it, so neither axis
            # may be stretched relative to the other.
            yaxis={
                "title_text": f"Axis {axis_y} ({share[axis_y - 1]:.1f}% of inertia)",
                "zeroline": True,
                "scaleanchor": "x",
                "scaleratio": 1,
            },
            width=setdict["html_aspect_ratio_w_over_h"] * setdict["html_image_height"],
            height=setdict["html_image_height"],
        )
        return fig


#: Eigenvalue corrections :class:`MCA` offers, by name.
MCA_CORRECTIONS = (None, "benzecri", "greenacre")


def _indicator(frame: pd.DataFrame, levels: dict[str, list] | None = None) -> pd.DataFrame:
    """One-hot encode every column, as ``"variable=level"``.

    With ``levels`` given, encode against those levels (the fitted ones), so a
    supplementary row lines up with the fitted categories; a level the fit never saw
    is refused rather than silently dropped, because dropping it would leave that
    row's profile summing to less than one and misplace it on the map.

    Raises
    ------
    ValueError
        If a cell is missing, or holds a level not in ``levels``.
    """
    if frame.isna().any().any():
        missing = frame.columns[frame.isna().any()].tolist()
        raise ValueError(
            f"Columns {missing} have missing values. MCA needs every observation to have a level in every "
            "variable; fill the gaps with an explicit level (for example 'unknown') if their absence "
            "is itself informative, or drop those rows."
        )
    blocks = []
    for column in frame.columns:
        values = frame[column].astype(str)
        known = sorted(values.unique()) if levels is None else levels[str(column)]
        unseen = sorted(set(values) - set(known))
        if unseen:
            raise ValueError(f"Column {column!r} has levels {unseen} that the model was not fitted on.")
        blocks.append(pd.DataFrame({f"{column}={level}": (values == level).astype(float) for level in known}))
    return pd.concat(blocks, axis=1).set_axis(frame.index, axis=0)


class MCA(CA):
    r"""Multiple correspondence analysis of several categorical variables.

    "PCA for categorical data": each observation is described by several categorical
    variables (grade, supplier, line, shift). MCA is correspondence analysis of the
    one-hot indicator matrix, so observations with similar combinations of levels
    plot together, and levels that tend to occur together plot together.

    The raw eigenvalues of an indicator matrix are notoriously pessimistic: with
    :math:`Q` variables, much of the inertia is an artefact of the coding, and the
    first axes look as if they explain a small share when they explain nearly all of
    the real association. ``correction`` reports that share honestly.

    Parameters
    ----------
    n_components : int, optional
        Number of axes to keep, default 2. At most :math:`J - Q` axes carry inertia,
        with :math:`J` the total number of levels.
    correction : {None, "benzecri", "greenacre"}, optional
        How to express each axis's share of the inertia:

        - ``None`` (default): the raw shares of the indicator matrix's inertia.
        - ``"benzecri"``: each eigenvalue above :math:`1/Q` becomes
          :math:`(Q/(Q-1))^2 (\lambda - 1/Q)^2`, the rest zero, and shares are taken of
          their sum. Known to be optimistic.
        - ``"greenacre"``: the same adjusted eigenvalues as a share of the adjusted
          total inertia :math:`\frac{Q}{Q-1}(\sum \lambda^2 - (J-Q)/Q^2)`, the sum taken
          over *every* axis. More conservative; the shares need not sum to 100%.

    Attributes
    ----------
    corrected_eigenvalues_ : np.ndarray of shape (n_components,)
        The eigenvalues after ``correction``; the raw ones when it is None.
    corrected_explained_inertia_ : np.ndarray of shape (n_components,)
        Share of the (corrected) inertia on each kept axis.
    variables_ : list[str]
        The categorical variables, in the order they were encoded.

    All the attributes of :class:`CA` are present as well. Its "rows" are the
    observations and its "columns" the levels, named ``"variable=level"``.

    References
    ----------
    M. Greenacre, "From simple to multiple correspondence analysis", in M. Greenacre
    and J. Blasius (eds.), Multiple Correspondence Analysis and Related Methods,
    Chapman and Hall/CRC, 2006.

    Examples
    --------
    >>> mca = MCA(n_components=2, correction="greenacre").fit(batch_attributes)  # doctest: +SKIP
    >>> mca.corrected_explained_inertia_                                         # doctest: +SKIP
    >>> mca.transform_columns(batch_attributes[["outcome"]])  # a supplementary label  # doctest: +SKIP
    """

    def __init__(self, n_components: int = 2, correction: str | None = None):
        super().__init__(n_components=n_components)
        self.correction = correction

    def fit(self, X: DataMatrix, y: object = None) -> MCA:  # noqa: ARG002
        """Fit to a table of categorical variables, one column per variable.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_observations, n_variables)
            Each column a categorical variable; values of any type are compared as
            strings.
        y : ignored
            Accepted for :class:`~sklearn.pipeline.Pipeline` compatibility.

        Returns
        -------
        MCA
            ``self``, fitted.

        Raises
        ------
        ValueError
            If ``correction`` is not recognised, fewer than two variables are given,
            or a cell is missing.
        """
        if self.correction not in MCA_CORRECTIONS:
            raise ValueError(f"correction must be one of {MCA_CORRECTIONS}; got {self.correction!r}.")
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        if frame.shape[1] < 2:
            raise ValueError(
                "MCA needs at least two categorical variables; with one, use CA on its cross-tabulation "
                "against another variable, or simply its frequency table."
            )
        indicator = _indicator(frame)
        super().fit(indicator)
        self.variables_ = [str(column) for column in frame.columns]
        self._levels = {str(column): sorted(frame[column].astype(str).unique()) for column in frame.columns}

        n_vars, n_levels = len(self.variables_), indicator.shape[1]
        raw = self.eigenvalues_
        if self.correction is None:
            self.corrected_eigenvalues_ = raw.copy()
            self.corrected_explained_inertia_ = self.explained_inertia_.copy()
            return self

        threshold = 1.0 / n_vars
        factor = (n_vars / (n_vars - 1)) ** 2
        self.corrected_eigenvalues_ = np.where(raw > threshold, factor * (raw - threshold) ** 2, 0.0)
        if self.correction == "benzecri":
            every = self._all_eigenvalues
            total = float(np.sum(np.where(every > threshold, factor * (every - threshold) ** 2, 0.0)))
        else:
            # The Burt matrix's inertia is the sum of squared eigenvalues over *every* axis.
            burt_inertia = float(np.sum(self._all_eigenvalues**2))
            total = n_vars / (n_vars - 1) * (burt_inertia - (n_levels - n_vars) / n_vars**2)
        # No eigenvalue above 1/Q means the variables are unassociated beyond what the
        # coding forces: every corrected eigenvalue is zero, and so is its total. That is
        # a finding, reported as zero shares, not 0/0.
        self.corrected_explained_inertia_ = (
            self.corrected_eigenvalues_ / total if total > 0 else np.zeros_like(self.corrected_eigenvalues_)
        )
        return self

    def transform(self, X: DataMatrix) -> pd.DataFrame:
        """Place observations on the fitted map, from their levels.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_observations, n_variables)
            The same variables as the fit, with levels the fit has seen.

        Returns
        -------
        pd.DataFrame
            Principal coordinates, one row per observation.

        Raises
        ------
        ValueError
            If the variables differ from the fit's, or a level is unseen or missing.
        """
        check_is_fitted(self, "variables_")
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        if [str(column) for column in frame.columns] != self.variables_:
            raise ValueError(f"Expected the variables {self.variables_}; got {list(frame.columns)}.")
        return super().transform(_indicator(frame, self._levels))

    def transform_columns(self, X: DataMatrix) -> pd.DataFrame:
        """Place the levels of *supplementary* categorical variables on the fitted map.

        Each level lands at the barycentre of the observations that have it, so a
        supplementary outcome label (good / bad batch) shows which combinations of the
        fitted attributes go with each outcome, without the outcome shaping the axes.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_observations, n_supplementary)
            One row per fitted observation, in the same order; one column per
            supplementary categorical variable.

        Returns
        -------
        pd.DataFrame
            Principal coordinates, one row per supplementary level, named
            ``"variable=level"``.
        """
        check_is_fitted(self, "variables_")
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        return super().transform_columns(_indicator(frame))
