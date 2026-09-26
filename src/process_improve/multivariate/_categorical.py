# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Latent-variable methods for categorical, count, mixed and grouped data (#176-#179).

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
from collections.abc import Mapping

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
        return _map_plot(self, axis_x, axis_y, settings, "Correspondence analysis map")


def _map_plot(model: CA | FAMD | MFA, axis_x: int, axis_y: int, settings: dict | None, default_title: str) -> go.Figure:
    """Plot a fitted model's row and column coordinates on one pair of axes, at a 1:1 ratio."""
    check_is_fitted(model, "row_coordinates_")

    class Settings(BaseModel):
        """Validated display settings for a categorical-methods map."""

        title: str = default_title
        row_color: str | None = None
        column_color: str | None = None
        html_image_height: float = 600.0
        html_aspect_ratio_w_over_h: float = 1.0
        template: str = DEFAULT_THEME

    setdict = Settings(**(settings or {})).model_dump()
    for axis in (axis_x, axis_y):
        if not 1 <= axis <= model.n_components_:
            raise ValueError(f"Axes run from 1 to {model.n_components_}; got {axis}.")

    fig = go.Figure()
    for frame, name, colour, symbol in (
        (model.row_coordinates_, "rows", setdict["row_color"], "circle"),
        (model.column_coordinates_, "columns", setdict["column_color"], "triangle-up"),
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
    share = 100 * model.explained_inertia_
    fig.update_layout(
        template=setdict["template"],
        title_text=setdict["title"],
        xaxis={"title_text": f"Axis {axis_x} ({share[axis_x - 1]:.1f}% of inertia)", "zeroline": True},
        # Equal scaling: distances on this map are the point of it, so neither axis may
        # be stretched relative to the other.
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


class FAMD(TransformerMixin, BaseEstimator):
    r"""Factor analysis of mixed data: one analysis of numeric and categorical columns together.

    Real process records mix continuous readings (temperature, pressure, flow) with
    categorical context (grade, supplier, line, shift). A PCA cannot take the
    categorical columns without one-hot encoding them by hand, after which they are
    over- or under-weighted against the numeric ones, because PCA has no notion of
    balancing the two. FAMD gives every variable an equal say: each numeric column
    is standardised, so it contributes an inertia of 1, and each categorical column
    with :math:`k` levels contributes :math:`k - 1`, exactly as it would in MCA.

    Concretely the numeric columns are centred and scaled to unit (population)
    variance, each level's indicator :math:`\delta` becomes
    :math:`\delta / \sqrt{p} - \sqrt{p}` for a level of frequency :math:`p`, and the
    combined matrix :math:`Z` is decomposed as :math:`Z / \sqrt{n} = U S V^T`. FAMD
    is exactly PCA of the correlation matrix when every column is numeric, and
    exactly MCA (eigenvalues scaled by :math:`Q`) when every column is categorical.

    Parameters
    ----------
    n_components : int, optional
        Number of axes to keep, default 2.
    categorical : list of str, optional
        Columns to treat as categorical even though they are numeric, such as a line
        or shift coded 1, 2, 3. Without it, numeric dtypes are numeric and everything
        else (strings, categoricals, booleans) is categorical.

    Attributes
    ----------
    n_components_ : int
        Axes actually kept.
    eigenvalues_ : np.ndarray of shape (n_components,)
        Inertia of each kept axis.
    total_inertia_ : float
        Numeric columns plus, for each categorical column, its levels minus one.
    explained_inertia_ : np.ndarray of shape (n_components,)
        Share of the total inertia on each kept axis.
    numeric_, categorical_ : list[str]
        How each column was treated.
    row_coordinates_ : pd.DataFrame
        One row per observation, one column per axis.
    column_coordinates_ : pd.DataFrame
        One row per numeric column and per categorical level (``"variable=level"``):
        for a numeric column, its correlation with the axis.
    row_contributions_, column_contributions_ : pd.DataFrame
        Share of each axis's inertia from each observation, and from each numeric
        column or level; each column sums to 1.
    row_cos2_ : pd.DataFrame
        Share of each observation's own inertia each axis represents.

    References
    ----------
    J. Pages, "Analyse factorielle de donnees mixtes", Revue de Statistique Appliquee,
    52 (2004), 93-111.

    Examples
    --------
    >>> famd = FAMD(n_components=2, categorical=["line"]).fit(batch_record)  # doctest: +SKIP
    >>> famd.explained_inertia_                                              # doctest: +SKIP
    >>> famd.map_plot()                                                      # doctest: +SKIP
    """

    def __init__(self, n_components: int = 2, categorical: list[str] | None = None):
        self.n_components = n_components
        self.categorical = categorical

    def _split(self, frame: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Decide which columns are numeric and which categorical."""
        forced = [str(c) for c in (self.categorical or [])]
        unknown = sorted(set(forced) - {str(c) for c in frame.columns})
        if unknown:
            raise ValueError(f"categorical names columns that are not in the data: {unknown}.")
        numeric_dtype = frame.select_dtypes(include="number").columns
        numeric = [str(c) for c in frame.columns if c in numeric_dtype and str(c) not in forced]
        categorical = [str(c) for c in frame.columns if str(c) not in numeric]
        return numeric, categorical

    def _transformed(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply the fitted standardisation and level weighting to ``frame``."""
        blocks = []
        if self.numeric_:
            values = frame[self.numeric_].to_numpy(dtype=float)
            blocks.append((values - self._means) / self._spreads)
        if self.categorical_:
            indicator = _indicator(frame[self.categorical_].astype(str), self._levels).to_numpy()
            blocks.append(indicator / np.sqrt(self._frequencies) - np.sqrt(self._frequencies))
        return np.hstack(blocks)

    def fit(self, X: DataMatrix, y: object = None) -> FAMD:  # noqa: ARG002
        """Fit to a table mixing numeric and categorical columns.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_observations, n_columns)
            The mixed table.
        y : ignored
            Accepted for :class:`~sklearn.pipeline.Pipeline` compatibility.

        Returns
        -------
        FAMD
            ``self``, fitted.

        Raises
        ------
        ValueError
            If ``n_components`` is not positive, a cell is missing, a numeric column
            is constant, or ``categorical`` names a column that is not there.
        """
        if int(self.n_components) < 1:
            raise ValueError(f"n_components must be at least 1; got {self.n_components}.")
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        frame = frame.set_axis([str(c) for c in frame.columns], axis=1)
        if frame.isna().any().any():
            raise ValueError(
                f"Columns {frame.columns[frame.isna().any()].tolist()} have missing values. Impute the numeric "
                "ones, give the categorical ones an explicit level, or drop those rows."
            )
        self.numeric_, self.categorical_ = self._split(frame)

        if self.numeric_:
            values = frame[self.numeric_].to_numpy(dtype=float)
            self._means = values.mean(axis=0)
            # Population variance, so that each standardised column carries an inertia of
            # exactly 1: that is what puts a numeric column on the same footing as a level.
            self._spreads = values.std(axis=0)
            constant = [name for name, s in zip(self.numeric_, self._spreads, strict=True) if s == 0]
            if constant:
                raise ValueError(f"Numeric columns {constant} are constant, so they carry no information. Drop them.")
        if self.categorical_:
            categories = frame[self.categorical_].astype(str)
            self._levels = {column: sorted(categories[column].unique()) for column in self.categorical_}
            self._frequencies = _indicator(categories, self._levels).to_numpy().mean(axis=0)

        transformed = self._transformed(frame)
        n_rows = transformed.shape[0]
        left, singular, right_t = np.linalg.svd(transformed / np.sqrt(n_rows), full_matrices=False)
        eigenvalues = singular**2
        rank = int(np.sum(eigenvalues > _NULL_INERTIA))
        if rank == 0:
            raise ValueError("The table has no variation to analyse: every observation is the same.")
        left, right = _flip_signs(left[:, :rank], right_t.T[:, :rank])

        keep = int(self.n_components)
        if keep > rank:
            warnings.warn(
                f"Asked for {keep} axes, but this table has {rank} with any inertia; keeping {rank}.",
                SpecificationWarning,
                stacklevel=2,
            )
            keep = rank

        column_names = [*self.numeric_, *[f"{c}={level}" for c in self.categorical_ for level in self._levels[c]]]
        axes = list(range(1, keep + 1))
        all_rows = transformed @ right
        self.n_components_ = keep
        self.eigenvalues_ = eigenvalues[:keep]
        self.total_inertia_ = float(eigenvalues.sum())
        self.explained_inertia_ = self.eigenvalues_ / self.total_inertia_
        self.row_coordinates_ = pd.DataFrame(all_rows[:, :keep], index=frame.index, columns=axes)
        self.column_coordinates_ = pd.DataFrame(right[:, :keep] * singular[:keep], index=column_names, columns=axes)
        self.row_contributions_ = pd.DataFrame(
            all_rows[:, :keep] ** 2 / n_rows / self.eigenvalues_, index=frame.index, columns=axes
        )
        self.column_contributions_ = pd.DataFrame(right[:, :keep] ** 2, index=column_names, columns=axes)
        self.row_cos2_ = pd.DataFrame(
            all_rows[:, :keep] ** 2 / np.maximum((all_rows**2).sum(axis=1, keepdims=True), np.finfo(float).tiny),
            index=frame.index,
            columns=axes,
        )
        self._loadings = right[:, :keep]
        self._columns = list(frame.columns)
        return self

    def transform(self, X: DataMatrix) -> pd.DataFrame:
        """Place new observations on the fitted map.

        Parameters
        ----------
        X : pd.DataFrame
            The same columns as the fit, with levels the fit has seen.

        Returns
        -------
        pd.DataFrame
            One row per observation, one column per axis.

        Raises
        ------
        ValueError
            If the columns differ from the fit's, or a level is unseen or missing.
        """
        check_is_fitted(self, "row_coordinates_")
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        frame = frame.set_axis([str(c) for c in frame.columns], axis=1)
        if list(frame.columns) != self._columns:
            raise ValueError(f"Expected the columns {self._columns}; got {list(frame.columns)}.")
        if frame.isna().any().any():
            raise ValueError("New observations must not have missing values.")
        return pd.DataFrame(
            self._transformed(frame) @ self._loadings, index=frame.index, columns=self.row_coordinates_.columns
        )

    def map_plot(self, axis_x: int = 1, axis_y: int = 2, settings: dict | None = None) -> go.Figure:
        """Draw observations and variables on one pair of axes.

        The column points are numeric columns (their correlations with the axes, so
        inside the unit circle) and categorical levels; the observations spread
        wider. Read a numeric column's direction, not its distance from an
        observation.

        Parameters
        ----------
        axis_x, axis_y : int, optional
            The axes to plot, counted from 1. Defaults 1 and 2.
        settings : dict, optional
            As for :meth:`CA.map_plot`, with the title defaulting to "FAMD map".

        Returns
        -------
        go.Figure
        """
        return _map_plot(self, axis_x, axis_y, settings, "FAMD map")


def _as_frame(X: DataMatrix | Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Return one frame from a frame, an array, or a dict of blocks that share their rows."""
    if not isinstance(X, Mapping):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    blocks = [pd.DataFrame(block) for block in X.values()]
    if any(not block.index.equals(blocks[0].index) for block in blocks):
        raise ValueError("Every block must have the same row index, in the same order.")
    frame = pd.concat(blocks, axis=1)
    repeated = frame.columns[frame.columns.duplicated()]
    if len(repeated):
        raise ValueError(f"Column names must be unique across blocks; {sorted(map(str, set(repeated)))} repeat.")
    return frame


class MFA(TransformerMixin, BaseEstimator):
    r"""Multiple factor analysis: one balanced analysis of several groups of variables.

    Process data arrives in blocks (a spectrum, a lab panel, the process conditions)
    measured on the same observations. Concatenating them into one PCA lets the widest
    block win: a 600-wavelength spectrum outvotes a dozen lab results simply by having
    more columns. MFA first analyses each group on its own and divides it by the square
    root of its first eigenvalue, so that no group's leading direction can carry more
    than an inertia of 1, then runs one PCA on the balanced whole.

    Two properties make the result readable. The first eigenvalue lies between 1 and
    the number of groups: near the number of groups means every block agrees on the
    dominant direction, near 1 means only one does. And each observation's position is
    the average of its *partial* positions, one per group, so the spread of an
    observation's partial points shows how much its blocks disagree about it.

    :class:`~process_improve.multivariate.methods.MBPCA` answers a related question
    with a different weighting: it gives each block the same *total* inertia, where MFA
    gives each block's *leading direction* the same inertia. The two agree exactly when
    every block's first eigenvalue is the same share of its total, as it is for
    one-dimensional blocks.

    Parameters
    ----------
    groups : dict[str, list[str]], optional
        Group name to the columns in it, when the data comes as one frame. Every
        column must be numeric and appear in exactly one group; columns in no group
        are ignored. Leave it out to pass the data as a dict of blocks instead, the
        form :class:`~process_improve.multivariate.methods.MBPCA` takes.
    n_components : int, optional
        Number of axes to keep, default 2.
    scale : bool, optional
        Scale every column to unit variance before the groups are formed, default
        True. Set False only when the columns within each group share units, a
        spectrum for instance, where scaling would inflate the noisy wavelengths.

    Attributes
    ----------
    n_components_ : int
        Axes actually kept.
    eigenvalues_ : np.ndarray of shape (n_components,)
        Inertia of each kept axis; the first lies between 1 and the number of groups.
    total_inertia_ : float
        Sum over every axis.
    explained_inertia_ : np.ndarray of shape (n_components,)
        Share of the total inertia on each kept axis.
    groups_ : dict[str, list[str]]
        The groups used: ``groups``, or the blocks' names and columns.
    group_weights_ : pd.Series
        The weight each group's columns were multiplied by: one over its first
        eigenvalue.
    row_coordinates_ : pd.DataFrame
        One row per observation, one column per axis.
    partial_row_coordinates_ : dict[str, pd.DataFrame]
        Per group, where that group alone would place each observation. Their average
        over groups is :attr:`row_coordinates_`.
    column_coordinates_ : pd.DataFrame
        Correlation of each column with each axis (for ``scale=True``).
    group_coordinates_ : pd.DataFrame
        One row per group: the inertia it contributes to each axis. Each lies between
        0 and 1, and a group near 1 on an axis has that axis as its own leading
        direction.
    column_contributions_, group_contributions_ : pd.DataFrame
        Share of each axis's inertia from each column, and from each group; each
        column sums to 1.

    References
    ----------
    B. Escofier and J. Pages, "Multiple factor analysis (AFMULT package)",
    Computational Statistics and Data Analysis, 18 (1994), 121-140.

    Examples
    --------
    >>> mfa = MFA({"spectra": wavelengths, "lab": assays}).fit(batches)   # doctest: +SKIP
    >>> mfa = MFA().fit({"spectra": spectra, "lab": lab})     # as for MBPCA   # doctest: +SKIP
    >>> mfa.eigenvalues_[0]        # near 2 means both blocks agree        # doctest: +SKIP
    >>> mfa.group_coordinates_                                            # doctest: +SKIP
    """

    def __init__(self, groups: dict[str, list[str]] | None = None, n_components: int = 2, scale: bool = True):
        self.groups = groups
        self.n_components = n_components
        self.scale = scale

    def _resolve_groups(self, X: DataMatrix | Mapping[str, pd.DataFrame]) -> dict[str, list[str]]:
        """Take the groups from a dict of blocks, or from ``groups``, but never from both."""
        if isinstance(X, Mapping):
            if self.groups is not None:
                raise ValueError("Pass either a dict of blocks or groups=, not both.")
            return {name: list(pd.DataFrame(block).columns) for name, block in X.items()}
        if self.groups is None:
            raise ValueError("Say which columns form each group with groups=, or pass a dict of blocks.")
        if not isinstance(self.groups, dict):
            raise TypeError(f"groups must be a dict of group name to column names; got {type(self.groups).__name__}.")
        return {name: list(columns) for name, columns in self.groups.items()}

    def _check_groups(self, frame: pd.DataFrame) -> None:
        """Refuse groups that overlap, name missing columns, hold non-numeric data, or are empty."""
        if len(self.groups_) < 2:
            raise ValueError("MFA needs at least two groups; with one, it is just a PCA.")
        seen: dict[str, str] = {}
        for name, columns in self.groups_.items():
            if not columns:
                raise ValueError(f"Group {name!r} has no columns.")
            for column in columns:
                if column not in frame.columns:
                    raise ValueError(f"Group {name!r} names column {column!r}, which is not in the data.")
                if column in seen:
                    raise ValueError(f"Column {column!r} is in both group {seen[column]!r} and group {name!r}.")
                seen[column] = name
                if not pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]):
                    raise ValueError(
                        f"Column {column!r} in group {name!r} is not numeric. MFA here takes numeric groups; "
                        "for a table mixing numeric and categorical columns use FAMD."
                    )

    def _group_block(self, frame: pd.DataFrame, name: str) -> np.ndarray:
        """Centre (and scale) one group's columns with the fitted statistics, then weight it."""
        values = frame[self.groups_[name]].to_numpy(dtype=float)
        centred = (values - self._means[name]) / self._spreads[name]
        return centred * np.sqrt(self.group_weights_[name])

    def _weigh_groups(self, frame: pd.DataFrame) -> None:
        """Store each group's centre and spread, and weight it by one over its first eigenvalue."""
        self._means, self._spreads = {}, {}
        weights = {}
        for name, columns in self.groups_.items():
            values = frame[columns].to_numpy(dtype=float)
            self._means[name] = values.mean(axis=0)
            spread = values.std(axis=0) if self.scale else np.ones(values.shape[1])
            if np.any(spread == 0):
                raise ValueError(f"Group {name!r} has a constant column; drop it before fitting.")
            self._spreads[name] = spread
            centred = (values - self._means[name]) / spread
            leading = np.linalg.svd(centred / np.sqrt(len(values)), compute_uv=False)[0] ** 2
            # Relative to the data's own size, so that an unscaled group measured in
            # tiny units is not mistaken for a constant one.
            if leading <= _NULL_INERTIA * np.mean(values**2):
                raise ValueError(f"Group {name!r} has no variation, so there is nothing to weight it by.")
            weights[name] = 1.0 / leading
        self.group_weights_ = pd.Series(weights, name="weight")

    def _per_group(self, blocks: dict[str, np.ndarray], index: pd.Index, axes: list[int]) -> None:
        """Split the global solution back into each group's partial points and inertia."""
        self.partial_row_coordinates_ = {}
        group_inertia, offset = {}, 0
        for name, block in blocks.items():
            rows = self._loadings[offset : offset + block.shape[1]]
            # Scaled by the number of groups so that the partial points average to the global one.
            self.partial_row_coordinates_[name] = pd.DataFrame(len(blocks) * block @ rows, index=index, columns=axes)
            group_inertia[name] = (rows**2).sum(axis=0) * self.eigenvalues_
            offset += block.shape[1]
        self.group_coordinates_ = pd.DataFrame(group_inertia, index=axes).T
        self.group_contributions_ = self.group_coordinates_ / self.eigenvalues_

    def fit(self, X: DataMatrix, y: object = None) -> MFA:  # noqa: ARG002
        """Fit to a table whose columns are organised into ``groups``.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_observations, n_columns), or dict[str, pd.DataFrame]
            One frame whose column names match those in ``groups``, or a dict of
            blocks with the same row index, one per group.
        y : ignored
            Accepted for :class:`~sklearn.pipeline.Pipeline` compatibility.

        Returns
        -------
        MFA
            ``self``, fitted.

        Raises
        ------
        ValueError
            If the groups are invalid or missing, the blocks do not share their rows,
            a cell is missing, ``n_components`` is not positive, or a group has no
            variation.
        TypeError
            If ``groups`` is given but is not a dict.
        """
        if int(self.n_components) < 1:
            raise ValueError(f"n_components must be at least 1; got {self.n_components}.")
        self.groups_ = self._resolve_groups(X)
        frame = _as_frame(X)
        self._check_groups(frame)
        used = [column for columns in self.groups_.values() for column in columns]
        if frame[used].isna().any().any():
            raise ValueError("MFA needs complete data; impute or drop the rows with missing values first.")

        self._weigh_groups(frame)
        blocks = {name: self._group_block(frame, name) for name in self.groups_}
        combined = np.hstack(list(blocks.values()))
        left, singular, right_t = np.linalg.svd(combined / np.sqrt(len(frame)), full_matrices=False)
        eigenvalues = singular**2
        rank = int(np.sum(eigenvalues > _NULL_INERTIA))
        left, right = _flip_signs(left[:, :rank], right_t.T[:, :rank])

        keep = int(self.n_components)
        if keep > rank:
            warnings.warn(
                f"Asked for {keep} axes, but this table has {rank} with any inertia; keeping {rank}.",
                SpecificationWarning,
                stacklevel=2,
            )
            keep = rank

        axes = list(range(1, keep + 1))
        self.n_components_ = keep
        self.eigenvalues_ = eigenvalues[:keep]
        self.total_inertia_ = float(eigenvalues.sum())
        self.explained_inertia_ = self.eigenvalues_ / self.total_inertia_
        self._loadings = right[:, :keep]
        self.row_coordinates_ = pd.DataFrame(combined @ self._loadings, index=frame.index, columns=axes)
        self._per_group(blocks, frame.index, axes)

        # A column's loading, undone of its group's weight and scaled to the axis's
        # spread: for standardised columns, its correlation with the axis.
        column_weights = self.group_weights_.to_numpy().repeat([len(c) for c in self.groups_.values()])
        self.column_coordinates_ = pd.DataFrame(
            self._loadings * singular[:keep] / np.sqrt(column_weights)[:, np.newaxis], index=used, columns=axes
        )
        self.column_contributions_ = pd.DataFrame(self._loadings**2, index=used, columns=axes)
        return self

    def transform(self, X: DataMatrix) -> pd.DataFrame:
        """Place new observations on the fitted map.

        Parameters
        ----------
        X : pd.DataFrame, or dict[str, pd.DataFrame]
            Observations with every column the groups name, in either of the forms
            :meth:`fit` takes.

        Returns
        -------
        pd.DataFrame
            One row per observation, one column per axis.

        Raises
        ------
        ValueError
            If a grouped column is absent or a cell is missing.
        """
        check_is_fitted(self, "row_coordinates_")
        frame = _as_frame(X)
        used = [column for columns in self.groups_.values() for column in columns]
        missing = [column for column in used if column not in frame.columns]
        if missing:
            raise ValueError(f"New observations lack the columns {missing}.")
        if frame[used].isna().any().any():
            raise ValueError("New observations must not have missing values.")
        combined = np.hstack([self._group_block(frame, name) for name in self.groups_])
        return pd.DataFrame(combined @ self._loadings, index=frame.index, columns=self.row_coordinates_.columns)

    def map_plot(self, axis_x: int = 1, axis_y: int = 2, settings: dict | None = None) -> go.Figure:
        """Draw the observations and the columns' correlations on one pair of axes.

        Parameters
        ----------
        axis_x, axis_y : int, optional
            The axes to plot, counted from 1. Defaults 1 and 2.
        settings : dict, optional
            As for :meth:`CA.map_plot`, with the title defaulting to "MFA map".

        Returns
        -------
        go.Figure
        """
        return _map_plot(self, axis_x, axis_y, settings, "MFA map")
