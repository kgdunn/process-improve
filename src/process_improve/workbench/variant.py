# (c) Kevin Dunn, 2010-2026. MIT License.
"""A model variant over a recipe's table: what is modelled, and which rows it leaves out.

A :class:`Variant` names the model (PCA or PLS), its number of components, the X and
Y columns, and the rows it **excludes**. Each block is mean-centred and scaled to unit
variance (:class:`~process_improve.multivariate.MCUVScaler`) on the fitted rows.
Excluding is not dropping: an excluded row is left out of the fit (its centring and
scaling too), then projected through the fitted model, so its scores, Hotelling's T²
and SPE are still reported and it can still be drawn. That is how a reader sees where
a suspect batch would fall in a model that never saw it.

:meth:`Variant.fit` returns a :class:`Fit`, and :meth:`Fit.explain_group` answers the
question a cluster on a score plot asks: what do these rows have in common that the
others do not?
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from process_improve.multivariate import PCA, PLS, MCUVScaler

from ._expressions import python_literal
from .recipe import RowId, _is_row_id, _text, _texts

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: The models a variant can be, by the name a project file stores.
MODELS: dict[str, type[PCA | PLS]] = {"PCA": PCA, "PLS": PLS}


@dataclass(frozen=True)
class Variant:
    """A model over a table: its kind, size, blocks, scaling, and excluded rows.

    Parameters
    ----------
    kind : {"PCA", "PLS"}
    n_components : int
    x_columns : tuple of str
        The X block. Empty takes every numeric column that is not in ``y_columns``.
    y_columns : tuple of str
        The Y block; PLS needs at least one column, PCA none.
    excluded : tuple of row ids
        Rows left out of the fit (and of its centring and scaling), then projected.
    name : str
        What the reader calls this variant.
    """

    kind: str = "PCA"
    n_components: int = 2
    x_columns: tuple[str, ...] = ()
    y_columns: tuple[str, ...] = ()
    excluded: tuple[RowId, ...] = ()
    name: str = "Model 1"

    def __post_init__(self) -> None:
        """Refuse a model kind that is not offered, a component count below one, and PLS without a Y."""
        if self.kind not in MODELS:
            raise ValueError(f"Unknown model {self.kind!r}; choose from {', '.join(MODELS)}.")
        if not isinstance(self.n_components, int) or isinstance(self.n_components, bool) or self.n_components < 1:
            raise ValueError(f"n_components must be a whole number of at least 1, got {self.n_components!r}.")
        if self.kind == "PLS" and not self.y_columns:
            raise ValueError("A PLS model needs at least one Y column.")

    def blocks(self, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Return the X block and the Y block (None for PCA) of ``table``, every row kept."""
        y_cols = list(self.y_columns)
        x_cols = list(self.x_columns) or [
            c for c in table.columns if c not in y_cols and pd.api.types.is_numeric_dtype(table[c])
        ]
        missing = [c for c in [*x_cols, *y_cols] if c not in table.columns]
        if missing:
            raise ValueError(f"The table has no column {', '.join(map(str, missing))}.")
        text = [c for c in [*x_cols, *y_cols] if not pd.api.types.is_numeric_dtype(table[c])]
        if text:
            raise ValueError(f"The columns {', '.join(map(str, text))} are not numbers; read them as numbers first.")
        x_block = table[x_cols].astype(float)
        return x_block, (table[y_cols].astype(float) if y_cols else None)

    def fit(self, table: pd.DataFrame) -> Fit:
        """Fit the variant to ``table``, then project its excluded rows through the model.

        Raises
        ------
        ValueError
            If an excluded row is not in the table, or too few rows remain to fit
            ``n_components`` components.
        """
        x_all, y_all = self.blocks(table)
        unknown = [r for r in self.excluded if r not in table.index]
        if unknown:
            raise ValueError(f"The excluded rows {', '.join(map(str, unknown))} are not in the table.")
        included = table.index.difference(pd.Index(list(self.excluded)), sort=False)
        if len(included) <= self.n_components:
            raise ValueError(
                f"{len(included)} rows remain after the exclusions, too few for {self.n_components} components."
            )
        scaler_x = MCUVScaler().fit(x_all.loc[included])
        x_scaled = scaler_x.transform(x_all)
        if self.kind == "PLS":
            y_block = cast("pd.DataFrame", y_all)  # __post_init__ refuses PLS without Y
            scaler_y = MCUVScaler().fit(y_block.loc[included])
            y_scaled = scaler_y.transform(y_block)
            # The blocks are already scaled on the fitted rows: PLS must not scale them again.
            model: PCA | PLS = PLS(n_components=self.n_components, scale=False).fit(
                x_scaled.loc[included], y_scaled.loc[included]
            )
        else:
            scaler_y, y_scaled = None, None
            model = PCA(n_components=self.n_components).fit(x_scaled.loc[included])
        return Fit(
            variant=self,
            table=table,
            model=model,
            x_scaled=x_scaled,
            y_scaled=y_scaled,
            scalers=(scaler_x, scaler_y),
        )

    def to_python(self, frame: str = "df") -> str:
        """Return the code that fits this variant to the table named ``frame``, as a recipe's script leaves it."""
        lines = [f"from process_improve.multivariate import {self.kind}, MCUVScaler", ""]
        if self.x_columns:
            lines.append(f"X = {frame}[{python_literal(list(self.x_columns))}].astype(float)")
        else:
            drop = f".drop(columns={python_literal(list(self.y_columns))})" if self.y_columns else ""
            lines.append(f'X = {frame}{drop}.select_dtypes("number").astype(float)')
        if self.y_columns:
            lines.append(f"Y = {frame}[{python_literal(list(self.y_columns))}].astype(float)")
        lines.append(f"excluded = {python_literal(list(self.excluded))}  # left out of the fit, projected afterwards")
        lines.append("included = X.index.difference(excluded, sort=False)")
        lines.append("scaler_x = MCUVScaler().fit(X.loc[included])")
        if self.kind == "PLS":
            lines.append("scaler_y = MCUVScaler().fit(Y.loc[included])")
            lines.append(
                f"model = PLS(n_components={self.n_components}, scale=False).fit("
                "scaler_x.transform(X.loc[included]), scaler_y.transform(Y.loc[included]))"
            )
        else:
            lines.append(f"model = PCA(n_components={self.n_components}).fit(scaler_x.transform(X.loc[included]))")
        if self.excluded:  # the scaler refuses an empty block, so a variant that excludes nothing projects nothing
            lines.append("projected = model.project(scaler_x.transform(X.loc[excluded]))  # scores, T2 and SPE")
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        """Return the variant as JSON-ready data."""
        out = dataclasses.asdict(self)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in out.items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Variant:
        """Rebuild a variant from :meth:`to_dict`'s output, checking every field."""
        excluded = data.get("excluded", [])
        if not isinstance(excluded, list | tuple) or not all(_is_row_id(v) for v in excluded):
            raise TypeError(f"'excluded' must be a list of row ids, got {excluded!r}.")
        n = data.get("n_components", 2)
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError(f"'n_components' must be a whole number, got {n!r}.")
        return cls(
            kind=_text(data, "kind", default="PCA"),
            n_components=n,
            x_columns=_texts({"x": data.get("x_columns", [])}, "x"),
            y_columns=_texts({"y": data.get("y_columns", [])}, "y"),
            excluded=tuple(excluded),
            name=_text(data, "name", default="Model 1"),
        )


@dataclass(frozen=True)
class Fit:
    """A fitted variant: the model, and every row's scores, T² and SPE, the excluded rows projected."""

    variant: Variant
    table: pd.DataFrame
    model: PCA | PLS
    x_scaled: pd.DataFrame
    y_scaled: pd.DataFrame | None
    scalers: tuple[MCUVScaler, MCUVScaler | None]

    @property
    def excluded(self) -> pd.Index:
        """The rows left out of the fit, in table order."""
        return self.table.index[self.table.index.isin(list(self.variant.excluded))]

    def rows(self) -> pd.DataFrame:
        """Return one row per table row: its scores, Hotelling's T² and SPE at all components, and ``excluded``.

        A fitted row carries the values the fit computed; an excluded row the values of
        projecting it through the model, its missing cells estimated from the observed
        ones (:meth:`PCA.project`).
        """
        model = self.model
        fitted = model.scores_.copy()
        fitted["T2"] = model.hotellings_t2_.iloc[:, -1].to_numpy()
        fitted["SPE"] = model.spe_.iloc[:, -1].to_numpy()
        frames = [fitted.assign(excluded=False)]
        if len(self.excluded):
            projected = model.project(self.x_scaled.loc[self.excluded])
            scores = pd.DataFrame(projected.scores.to_numpy(), index=self.excluded, columns=model.scores_.columns)
            scores["T2"] = projected.hotellings_t2.to_numpy()
            scores["SPE"] = projected.spe.to_numpy()
            frames.append(scores.assign(excluded=True))
        return pd.concat(frames).reindex(self.table.index)

    def r2x(self) -> list[float]:
        """Return the fraction of the fitted X block's variance each component explains, in order."""
        x_fit = self.x_scaled.loc[self.model.scores_.index].to_numpy()
        total = float(np.nansum(x_fit**2))
        residual = [
            float(np.nansum(self.model.spe_.iloc[:, k].to_numpy() ** 2)) for k in range(self.model.n_components_)
        ]
        cumulative = [1.0 - r / total for r in residual]
        return [cumulative[0], *np.diff(cumulative).tolist()]

    def limits(self, conf_level: float = 0.95) -> dict[str, float]:
        """Return the T² and SPE limits at ``conf_level``."""
        return {
            "T2": float(self.model.hotellings_t2_limit(conf_level)),
            "SPE": float(self.model.spe_limit(conf_level=conf_level)),
        }

    def explain_group(
        self,
        group: Sequence[RowId],
        reference: Sequence[RowId] | None = None,
        components: Sequence[int] | None = None,
        top: int = 5,
    ) -> GroupExplanation:
        """Explain why ``group`` sits where it does: which variables move its average score away from the reference's.

        Parameters
        ----------
        group : sequence of row ids
            The rows of interest, for example a lasso on a score plot. Excluded rows may
            be part of it.
        reference : sequence of row ids, optional
            The rows to compare against. Default: every fitted row outside the group.
        components : sequence of int, optional
            1-based components to explain; default all.
        top : int, default 5
            How many variables :attr:`GroupExplanation.top` names.

        Returns
        -------
        GroupExplanation
        """
        index = self.table.index
        group_ids = _rows_in(index, group, "group")
        if reference is None:
            fitted = self.model.scores_.index
            reference_ids = fitted[~fitted.isin(group_ids)]
        else:
            reference_ids = _rows_in(index, reference, "reference")
        if len(reference_ids) == 0:
            raise ValueError("The reference holds no rows: there is nothing to compare the group against.")
        chosen = list(components) if components is not None else list(range(1, self.variant.n_components + 1))
        x = _filled(self)
        contributions = pd.DataFrame(
            {
                f"PC{a}": self.model.group_contributions(x, group=group_ids, reference=reference_ids, component=a)
                for a in chosen
            }
        )
        biggest = contributions.abs().max(axis=1).sort_values(ascending=False)
        top_variables = list(biggest.index[: max(0, top)])
        raw, _ = self.variant.blocks(self.table)
        g, r = raw.loc[group_ids], raw.loc[reference_ids]
        spread = r.std(ddof=1).replace(0.0, np.nan)
        summary = pd.DataFrame(
            {
                "group mean": g.mean(),
                "reference mean": r.mean(),
                "difference": g.mean() - r.mean(),
                "difference in reference SDs": (g.mean() - r.mean()) / spread,
            }
        ).loc[biggest.index]
        return GroupExplanation(
            group=list(group_ids),
            reference=list(reference_ids),
            contributions=contributions.loc[biggest.index],
            top=top_variables,
            summary=summary,
            evidence=raw[top_variables].assign(in_group=index.isin(group_ids), in_reference=index.isin(reference_ids)),
        )


@dataclass(frozen=True)
class GroupExplanation:
    """Why a group of rows differs from a reference, variable by variable.

    Attributes
    ----------
    group, reference : list of row ids
    contributions : pd.DataFrame
        One row per variable, one column per component: each variable's share of the
        difference between the group's average score and the reference's. A column
        sums to that difference. Rows are ordered by their largest absolute share.
    top : list of str
        The variables with the largest shares, the ones to look at first.
    summary : pd.DataFrame
        Per variable, in the same order, the group's and the reference's means in the
        table's own units, their difference, and the difference in units of the
        reference's standard deviation.
    evidence : pd.DataFrame
        The raw values of the top variables for every row, in table order, with
        ``in_group`` and ``in_reference`` flags: what to draw beside the bars.
    """

    group: list[RowId]
    reference: list[RowId]
    contributions: pd.DataFrame
    top: list[str]
    summary: pd.DataFrame
    evidence: pd.DataFrame

    def score_difference(self) -> pd.Series:
        """Return, per component, the difference between the group's average score and the reference's."""
        return self.contributions.sum()


def _rows_in(index: pd.Index, rows: Sequence[RowId], name: str) -> pd.Index:
    selected = index[index.isin(list(rows))]
    unknown = [r for r in rows if r not in index]
    if unknown:
        raise ValueError(f"The {name} names rows that are not in the table: {', '.join(map(str, unknown[:5]))}.")
    if len(selected) == 0:
        raise ValueError(f"The {name} holds no rows.")
    return selected


def _filled(fit: Fit) -> pd.DataFrame:
    """Return the scaled X block with each missing cell replaced by the model's estimate of it.

    The group's average of a column with a missing cell would otherwise be missing,
    and so would every contribution. A fitted row is reconstructed from its scores, an
    excluded one from its projected scores: ``x_hat = t P'`` with the X loadings.
    """
    x = fit.x_scaled
    if not x.isna().to_numpy().any():
        return x
    rows = fit.rows()
    scores = rows.iloc[:, : fit.variant.n_components].to_numpy()
    loadings = fit.model.loadings_ if isinstance(fit.model, PCA) else fit.model.x_loadings_
    estimate = pd.DataFrame(scores @ loadings.to_numpy().T, index=x.index, columns=x.columns)
    return x.fillna(estimate)
