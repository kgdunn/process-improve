"""Generalized Procrustes analysis: one consensus from several configurations of the same objects (#180).

A panel of assessors scores the same products, each on their own use of the scale;
two instruments measure the same parts, each in its own frame. Every one of them gives
a *configuration*: the objects as points in space. Configurations can differ in where
they are centred, how they are oriented (mirrored included), and how large they are,
and none of that says anything about the objects. GPA removes exactly those three
differences, and what is left is agreement (the consensus) and disagreement (the
residuals), object by object and configuration by configuration.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

from process_improve._random import check_random_state
from process_improve.visualization.themes import DEFAULT_THEME

from ._common import SpecificationWarning

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - exercised via env-without-plotly
    from process_improve._extras import _MissingExtra

    go = _MissingExtra("plotly", "plotting")  # type: ignore[assignment]


#: Centred spread, as a share of a configuration's uncentred size, below which every
#: object is taken to sit at the same point.
_NULL_SPREAD = 1e-12


def _rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the orthogonal matrix (reflections allowed) that takes ``source`` closest to ``target``."""
    left, _, right_t = np.linalg.svd(source.T @ target)
    return left @ right_t


def _optimal_scaling(rotated: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return ten Berge's (1977) scaling factors for unit-size configurations, and their signs.

    For a fixed total size, the consensus is largest (and the residual smallest) when the
    factors follow the leading eigenvector of the configurations' inner products. A
    negative entry means that configuration fits better turned inside out; that is an
    orthogonal transformation too, so its sign is returned for the rotation to absorb.
    """
    inner = np.array([[np.sum(a * b) for b in rotated] for a in rotated])
    _, vectors = np.linalg.eigh(inner)
    leading = vectors[:, -1] * (1.0 if vectors[:, -1].sum() >= 0 else -1.0)
    signs = np.where(leading < 0, -1.0, 1.0)
    return np.sqrt(len(rotated)) * np.abs(leading), signs


def _residual_ss(aligned: list[np.ndarray]) -> float:
    """Return the summed squared distance of every aligned configuration from their mean."""
    consensus = sum(aligned) / len(aligned)
    return float(sum(np.sum((config - consensus) ** 2) for config in aligned))


def _consensus_share(aligned: list[np.ndarray]) -> float:
    """Return the share of the aligned configurations' total sum of squares that their mean accounts for."""
    consensus = sum(aligned) / len(aligned)
    return float(len(aligned) * np.sum(consensus**2) / sum(np.sum(config**2) for config in aligned))


def _align(configs: list[np.ndarray], *, scale: bool, tol: float, max_iter: int) -> Bunch:
    """Alternate rotations and (optionally) scalings until the residual sum of squares stops falling.

    Each configuration in turn is rotated onto the sum of all the others (Gower, 1975),
    which cannot increase the residual; so cannot the scaling step. The loop therefore
    descends monotonically, and stops once a round gains less than ``tol`` of the total.
    """
    count = len(configs)
    rotations = [np.eye(configs[0].shape[1]) for _ in configs]
    factors = np.ones(count)
    total = sum(float(np.sum(config**2)) for config in configs)
    loss, converged, n_iter = np.inf, False, 0
    while n_iter < max_iter and not converged:
        n_iter += 1
        for k in range(count):
            others = sum(factors[j] * configs[j] @ rotations[j] for j in range(count) if j != k)
            rotations[k] = _rotation(configs[k], others)
        if scale:
            factors, signs = _optimal_scaling(
                [config @ rotation for config, rotation in zip(configs, rotations, strict=True)]
            )
            rotations = [rotation * sign for rotation, sign in zip(rotations, signs, strict=True)]
        aligned = [
            factor * config @ rotation for factor, config, rotation in zip(factors, configs, rotations, strict=True)
        ]
        previous, loss = loss, _residual_ss(aligned)
        converged = previous - loss <= tol * total
    return Bunch(aligned=aligned, rotations=rotations, factors=factors, loss=loss, n_iter=n_iter, converged=converged)


def _check_configurations(X: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Return the configurations as frames, refusing any that do not describe the same objects."""
    if not isinstance(X, Mapping):
        raise TypeError(
            f"GPA takes a dict of configurations, name to DataFrame (as MBPCA takes blocks); got {type(X).__name__}."
        )
    frames = {str(name): pd.DataFrame(config) for name, config in X.items()}
    if len(frames) < 2:
        raise ValueError("GPA needs at least two configurations to compare.")
    first = next(iter(frames.values()))
    for name, frame in frames.items():
        if not frame.index.equals(first.index):
            raise ValueError(f"Configuration {name!r} does not have the same objects, in the same order, as the first.")
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in frame.dtypes):
            raise ValueError(f"Configuration {name!r} has non-numeric columns.")
        if frame.isna().to_numpy().any():
            raise ValueError(f"Configuration {name!r} has missing values; impute or drop those objects first.")
    return frames


def _padded(values: np.ndarray, width: int) -> np.ndarray:
    """Pad with zero columns to ``width``, so configurations of different dimension can be rotated together."""
    return np.hstack([values, np.zeros((values.shape[0], width - values.shape[1]))])


class GPA(BaseEstimator):
    r"""Generalized Procrustes analysis: a consensus from several configurations of the same objects.

    Each configuration places the same objects (products, parts, batches) as points,
    one column per dimension: an assessor's scores on their attributes, an instrument's
    measurements. GPA centres every configuration, rotates it (reflections allowed)
    and, with ``scale=True``, stretches or shrinks it as a whole, so that together they
    sit as close as possible to their mean, the *consensus* (Gower, 1975; ten Berge,
    1977). The configurations may have different numbers of columns, as in free-choice
    profiling where each assessor uses their own words; the narrower ones are padded
    with zeros.

    GPA always finds a consensus, even among unrelated configurations: four of pure
    noise typically share around half their variation after alignment. Judge the
    consensus with :meth:`consensus_test`, not with :attr:`consensus_share_` alone.

    Parameters
    ----------
    scale : bool, optional
        Give each configuration its own isotropic scaling factor, default True. Keep it
        for panels, where assessors use different ranges of the scale; set False when
        the configurations share units and a difference in size is itself a finding.
    tol : float, optional
        Stop when a round of rotations and scalings lowers the residual sum of squares
        by less than this share of the total, default 1e-10.
    max_iter : int, optional
        Most rounds to run, default 100. Convergence usually takes under twenty.

    Attributes
    ----------
    consensus_ : pd.DataFrame of shape (n_objects, n_dimensions)
        The mean of the aligned configurations, turned to its principal axes, so the
        first column carries the most variation.
    aligned_ : dict[str, pd.DataFrame]
        Each configuration after translation, rotation and scaling, in the consensus's
        axes.
    scale_factors_ : pd.Series
        The factor each centred configuration was multiplied by. Above 1, it was
        stretched: the assessor used a narrower range than the others. The factors
        preserve the total sum of squares of the centred configurations.
    rotations_ : dict[str, np.ndarray]
        The orthogonal matrix, of shape (n_dimensions, n_dimensions), applied to each
        centred and padded configuration.
    residuals_ : pd.DataFrame of shape (n_objects, n_configurations)
        Squared distance of each object from the consensus, in each configuration.
        Summed down a column, how far a configuration disagrees; along a row, how
        contested an object is.
    consensus_share_ : float
        The share of the aligned configurations' total sum of squares that the
        consensus accounts for: one when all agree exactly.
    explained_variance_ratio_ : np.ndarray of shape (n_dimensions,)
        The share of the consensus's own variation on each of its axes.
    n_dimensions_ : int
        Columns in the widest configuration.
    n_iter_ : int
        Rounds run.

    References
    ----------
    J.C. Gower, "Generalized Procrustes analysis", Psychometrika, 40 (1975), 33-51.

    J.M.F. ten Berge, "Orthogonal Procrustes rotation for two or more matrices",
    Psychometrika, 42 (1977), 267-276.

    G.M. Arnold and A.A. Williams, "The use of generalised Procrustes techniques in
    sensory analysis", in *Statistical Procedures in Food Research*, Elsevier (1986).

    I.N. Wakeling, M.M. Raats and H.J.H. MacFie, "A new significance test for consensus
    in generalized Procrustes analysis", Journal of Sensory Studies, 7 (1992), 91-96.

    Examples
    --------
    >>> panel_configs = GPA.configurations_from_long(panel)       # doctest: +SKIP
    >>> gpa = GPA().fit(panel_configs)                           # doctest: +SKIP
    >>> gpa.residuals_.sum().sort_values()      # who disagrees  # doctest: +SKIP
    >>> gpa.consensus_test(random_state=0).p_value               # doctest: +SKIP
    """

    def __init__(self, scale: bool = True, tol: float = 1e-10, max_iter: int = 100):
        self.scale = scale
        self.tol = tol
        self.max_iter = max_iter

    @staticmethod
    def configurations_from_long(
        frame: pd.DataFrame,
        *,
        configuration: str = "panelist_id",
        objects: str = "product",
        variables: str = "attribute",
        values: str = "score",
    ) -> dict[str, pd.DataFrame]:
        """Pivot a long table into one configuration per ``configuration`` value: objects by variables.

        Values are averaged over everything else, such as replicates and sessions. The
        defaults are the column names of the sensory subpackage's ``descriptive_long``
        schema, so a validated panel goes straight in. A configuration keeps only the
        variables it used, as free-choice profiling needs.

        Parameters
        ----------
        frame : pd.DataFrame
            One row per value.
        configuration, objects, variables, values : str, optional
            The columns naming the configuration (the assessor), the object (the
            product), the variable (the attribute) and holding the value (the score).

        Returns
        -------
        dict[str, pd.DataFrame]
            Configuration name to a table of objects by variables, ready for :meth:`fit`.

        Raises
        ------
        ValueError
            If a named column is missing.
        """
        needed = [configuration, objects, variables, values]
        missing = [column for column in needed if column not in frame.columns]
        if missing:
            raise ValueError(f"The table lacks the columns {missing}.")
        table = frame.pivot_table(index=[configuration, objects], columns=variables, values=values, aggfunc="mean")
        return {
            str(name): group.droplevel(0).dropna(axis=1, how="all")
            for name, group in table.groupby(level=0, sort=False)
        }

    def fit(self, X: Mapping[str, pd.DataFrame], y: object = None) -> GPA:  # noqa: ARG002
        """Align the configurations and find their consensus.

        Parameters
        ----------
        X : dict[str, pd.DataFrame]
            Configuration name to a table with one row per object and one column per
            dimension. Every table must have the same objects, in the same order, as
            its index; the columns may differ.
        y : ignored
            Accepted for :class:`~sklearn.pipeline.Pipeline` compatibility.

        Returns
        -------
        GPA
            ``self``, fitted.

        Raises
        ------
        TypeError
            If ``X`` is not a dict of tables.
        ValueError
            If there are fewer than two configurations or three objects, the objects
            differ between configurations, a value is missing or not numeric, or a
            configuration puts every object at the same point.

        Warns
        -----
        SpecificationWarning
            If ``max_iter`` rounds end before the residual settles.
        """
        frames = _check_configurations(X)
        if len(next(iter(frames.values()))) < 3:
            raise ValueError(
                "GPA needs at least three objects: two points always lie on a line, and any two lines "
                "can be turned onto each other, so there would be nothing to compare."
            )
        for name, frame in frames.items():
            values = frame.to_numpy(dtype=float)
            # Relative to the data's own size, so tiny units are not mistaken for no spread.
            if np.linalg.norm(values - values.mean(axis=0)) <= _NULL_SPREAD * np.linalg.norm(values):
                raise ValueError(f"Configuration {name!r} puts every object at the same point.")
        self.n_dimensions_ = max(frame.shape[1] for frame in frames.values())
        self._centres = {name: frame.mean().to_numpy(dtype=float) for name, frame in frames.items()}
        self._columns = {name: list(frame.columns) for name, frame in frames.items()}
        centred = [
            _padded(frame.to_numpy(dtype=float) - self._centres[name], self.n_dimensions_)
            for name, frame in frames.items()
        ]
        sizes = np.array([np.linalg.norm(config) for config in centred])
        # Unit size for the iterations when scaling, so the factors start equal.
        self._configs = [config / size for config, size in zip(centred, sizes, strict=True)] if self.scale else centred
        fitted = _align(self._configs, scale=self.scale, tol=self.tol, max_iter=self.max_iter)
        if not fitted.converged:
            warnings.warn(
                f"GPA stopped after max_iter={self.max_iter} rounds before the residual settled; raise max_iter.",
                SpecificationWarning,
                stacklevel=2,
            )
        self.n_iter_ = fitted.n_iter
        # Back to the data's own size: the aligned configurations keep the centred total sum of squares.
        factors = fitted.factors * np.sqrt(np.mean(sizes**2)) / sizes if self.scale else np.ones(len(frames))
        self._finish(frames, centred, factors, fitted.rotations)
        return self

    def _finish(
        self,
        frames: dict[str, pd.DataFrame],
        centred: list[np.ndarray],
        factors: np.ndarray,
        rotations: list[np.ndarray],
    ) -> None:
        """Turn everything to the consensus's principal axes and store the fitted results."""
        objects = next(iter(frames.values())).index
        dimensions = list(range(1, self.n_dimensions_ + 1))
        consensus = sum(f * c @ r for f, c, r in zip(factors, centred, rotations, strict=True)) / len(frames)
        _, singular, axes_t = np.linalg.svd(consensus)
        axes = axes_t.T
        # Deterministic signs: each axis points towards the object furthest along it.
        turned = consensus @ axes
        axes *= np.where(turned[np.abs(turned).argmax(axis=0), range(axes.shape[1])] < 0, -1.0, 1.0)

        self.scale_factors_ = pd.Series(factors, index=list(frames), name="scale factor")
        self.rotations_ = {name: rotation @ axes for name, rotation in zip(frames, rotations, strict=True)}
        aligned = {
            name: factor * config @ self.rotations_[name]
            for name, factor, config in zip(frames, factors, centred, strict=True)
        }
        self.aligned_ = {
            name: pd.DataFrame(values, index=objects, columns=dimensions) for name, values in aligned.items()
        }
        self.consensus_ = pd.DataFrame(sum(aligned.values()) / len(aligned), index=objects, columns=dimensions)
        self.residuals_ = pd.DataFrame(
            {name: ((values - self.consensus_.to_numpy()) ** 2).sum(axis=1) for name, values in aligned.items()},
            index=objects,
        )
        self.consensus_share_ = _consensus_share(list(aligned.values()))
        spread = np.zeros(self.n_dimensions_)
        spread[: len(singular)] = singular**2
        self.explained_variance_ratio_ = spread / spread.sum()

    def transform(self, X: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Place new objects in the consensus, through each configuration's fitted alignment.

        Parameters
        ----------
        X : dict[str, pd.DataFrame]
            The same configurations, with the same columns, describing new objects
            (one or more rows each, the same objects in every configuration).

        Returns
        -------
        pd.DataFrame
            One row per object: the mean of where each configuration puts it.

        Raises
        ------
        ValueError
            If a configuration is missing or unexpected, or its columns differ from the fit.
        """
        check_is_fitted(self, "consensus_")
        frames = _check_configurations(X)
        if set(frames) != set(self.rotations_):
            raise ValueError(f"Expected the configurations {sorted(self.rotations_)}; got {sorted(frames)}.")
        positions = []
        for name, frame in frames.items():
            if list(frame.columns) != self._columns[name]:
                raise ValueError(
                    f"Configuration {name!r} has columns {list(frame.columns)}; expected {self._columns[name]}."
                )
            centred = _padded(frame.to_numpy(dtype=float) - self._centres[name], self.n_dimensions_)
            positions.append(self.scale_factors_[name] * centred @ self.rotations_[name])
        objects = next(iter(frames.values())).index
        return pd.DataFrame(sum(positions) / len(positions), index=objects, columns=self.consensus_.columns)

    def consensus_test(
        self, n_permutations: int = 199, *, random_state: int | np.random.Generator | None = None
    ) -> Bunch:
        """Test whether the consensus is more than GPA would find among unrelated configurations.

        The objects are shuffled independently within every configuration but the
        first, which destroys any agreement about which object is which while keeping
        each configuration's own shape, and GPA is refitted. The p-value is the share
        of shuffles whose consensus accounts for at least as much of the total as the
        real one (Wakeling, Raats and MacFie, 1992).

        Parameters
        ----------
        n_permutations : int, optional
            Shuffles to run, default 199; the smallest p-value possible is one over
            this plus one.
        random_state : int, np.random.Generator or None, optional
            Seed or generator for the shuffles; None draws fresh ones.

        Returns
        -------
        Bunch
            ``consensus_share`` (the fitted one), ``null_shares`` (one per shuffle) and
            ``p_value``.

        Raises
        ------
        ValueError
            If ``n_permutations`` is below 1.
        """
        check_is_fitted(self, "consensus_")
        if int(n_permutations) < 1:
            raise ValueError(f"n_permutations must be at least 1; got {n_permutations}.")
        rng = check_random_state(random_state)
        null = np.empty(int(n_permutations))
        for index in range(len(null)):
            shuffled = [self._configs[0]] + [config[rng.permutation(len(config))] for config in self._configs[1:]]
            refit = _align(shuffled, scale=self.scale, tol=self.tol, max_iter=self.max_iter)
            null[index] = _consensus_share(refit.aligned)
        p_value = (1 + int(np.sum(null >= self.consensus_share_))) / (1 + len(null))
        return Bunch(consensus_share=self.consensus_share_, null_shares=null, p_value=p_value)

    def map_plot(self, axis_x: int = 1, axis_y: int = 2, settings: dict | None = None) -> go.Figure:
        """Draw the consensus, with a line from each object to where each configuration puts it.

        Short spokes mean agreement; one configuration's long spokes point at the
        assessor or instrument that sees the objects differently.

        Parameters
        ----------
        axis_x, axis_y : int, optional
            The consensus axes to plot, counted from 1. Defaults 1 and 2.
        settings : dict, optional
            ``title`` (default "GPA consensus"), ``html_image_height`` (default 600),
            ``html_aspect_ratio_w_over_h`` (default 1) and ``template``.

        Returns
        -------
        go.Figure

        Raises
        ------
        ValueError
            If an axis is outside 1 to ``n_dimensions_``.
        """
        check_is_fitted(self, "consensus_")

        class Settings(BaseModel):
            """Validated display settings for the GPA consensus map."""

            title: str = "GPA consensus"
            html_image_height: float = 600.0
            html_aspect_ratio_w_over_h: float = 1.0
            template: str = DEFAULT_THEME

        setdict = Settings(**(settings or {})).model_dump()
        for axis in (axis_x, axis_y):
            if not 1 <= axis <= self.n_dimensions_:
                raise ValueError(f"Axes run from 1 to {self.n_dimensions_}; got {axis}.")
        centre = self.consensus_[[axis_x, axis_y]].to_numpy()
        fig = go.Figure()
        for name, aligned in self.aligned_.items():
            ends = aligned[[axis_x, axis_y]].to_numpy()
            # One trace per configuration: a spoke per object, separated by gaps.
            spokes = np.stack([centre, ends, np.full_like(centre, np.nan)], axis=1).reshape(-1, 2)
            fig.add_trace(go.Scatter(x=spokes[:, 0], y=spokes[:, 1], mode="lines", name=name, opacity=0.6))
        fig.add_trace(
            go.Scatter(
                x=centre[:, 0],
                y=centre[:, 1],
                mode="markers+text",
                text=[str(label) for label in self.consensus_.index],
                textposition="top center",
                name="consensus",
                marker={"size": 10, "color": "black"},
            )
        )
        share = 100 * self.explained_variance_ratio_
        fig.update_layout(
            template=setdict["template"],
            title_text=setdict["title"],
            xaxis={"title_text": f"Axis {axis_x} ({share[axis_x - 1]:.1f}% of the consensus)", "zeroline": True},
            yaxis={
                "title_text": f"Axis {axis_y} ({share[axis_y - 1]:.1f}% of the consensus)",
                "zeroline": True,
                "scaleanchor": "x",
                "scaleratio": 1,
            },
            width=setdict["html_aspect_ratio_w_over_h"] * setdict["html_image_height"],
            height=setdict["html_image_height"],
        )
        return fig
