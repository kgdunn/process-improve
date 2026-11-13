# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.
"""Fitting a linear model to an experiment: :func:`lm`, the :class:`Model` it returns, and its summary.

The helpers that read, check and rewrite a formula live in :mod:`process_improve.experiments._formula`, which
needs no statsmodels; they are offered from here too, as they always were.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import ModelDesc
from statsmodels.regression.linear_model import OLS

from process_improve.experiments._formula import (
    UnsafeFormulaError,
    forg,
    real_names,
    stand_ins_for,
    suggest_identifier,
    validate_contrast_levels,
    validate_formula_is_safe,
    validate_identifier_is_safe,
    validate_identifiers_are_safe,
    with_stand_ins,
)

if TYPE_CHECKING:
    from process_improve.experiments.structures import Expt


class Model(OLS):
    """Just a thin wrapper around the OLS class from Statsmodels."""

    # Declared for static typing. ``data`` starts as ``None`` and is replaced by
    # the fitted :class:`~process_improve.experiments.structures.Expt` in ``lm()``.
    data: Expt | None
    aliasing: dict | None
    name: str | None
    #: Each column fitted under a stand-in name, mapped to it (see :func:`stand_ins_for`); set by ``lm()``.
    _stand_ins: dict[str, str]

    def __init__(
        self,
        OLS_instance: Any,  # noqa: ANN401
        model_spec: str,
        aliasing: dict | None = None,
        name: str | None = None,
    ) -> None:
        self._OLS = OLS_instance
        self._model_spec = model_spec
        self.name = name

        # Standard error
        self.df_resid = self._OLS.df_resid
        self.df_model = self._OLS.df_model
        self.nobs = self._OLS.nobs
        # Leads to errors for size inconsistency if the data frames have
        # missing data?
        # self.rsquared = self.R2 = self._OLS.rsquared
        self.residuals = self._OLS.resid

        # Will be replaced by the "lm()" function
        self.data = None
        self.aliasing = aliasing
        self._stand_ins = {}

    def __str__(self) -> str:
        """Return the model specification as a string."""
        spec = ModelDesc.from_formula(self._model_spec)
        return spec.describe()

    def summary(self, alpha: float = 0.05, print_to_screen: bool = True) -> Any:  # noqa: ARG002, ANN401
        """Build the OLS summary table for this model and return it.

        The returned object is the statsmodels summary instance, with the
        underlying ``self._OLS.summary()`` adjusted to label the residual
        standard error row and to leave out the date and time of the call,
        so that the same fit gives the same summary. The method does NOT
        print anything by itself;
        the top-level :func:`summary` wrapper handles screen output via its
        own ``show`` flag. The ``alpha`` and ``print_to_screen`` arguments
        are unused and kept for backwards compatibility.
        """
        # Taken from statsmodels.regression.linear_model.py
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            main = "OLS Regression Results"
            title = self.name or self.get_title()
            if title:
                main += ": " + str(title)

            smry = self._OLS.summary(title=main)
            # print(smry)
            # Call this directly and modify the result to suppress what we
            # don't really care to show:

            smry.tables[0].pop(8)

            se = "---"
            if not (np.isinf(self._OLS.scale)):
                se = f"{np.sqrt(self._OLS.scale):.3f}"

            # Residual standard error
            smry.tables[0][7][0].data = "Residual std error"
            smry.tables[0][7][1].data = se
            # smry.tables[0][7][0].data = se
            # smry.tables[0][7][1].data = se

            # The time of the call is no fact about the model, and it made two summaries of one fit differ
            # (#952): leave out the Date and Time cells, move the left-hand cells below them up, and drop
            # the row that leaves empty.
            top = smry.tables[0]
            left = [(row[0].data, row[1].data) for row in top if row[0].data not in ("Date:", "Time:")]
            for row, (label, value) in zip(top, left + [("", "")] * (len(top) - len(left)), strict=True):
                row[0].data, row[1].data = label, value
            while not "".join(str(cell.data) for cell in top[-1]).strip():
                top.pop()

        return smry

    def get_parameters(self, drop_intercept: bool = True) -> pd.Series:
        """Get the parameter values; return them in a pandas Series indexed by term."""

        params = self._OLS.params.copy()
        try:
            if drop_intercept:
                params = params.drop("Intercept")
        except KeyError:
            # Some models (e.g. ``y ~ 0 + ...``) have no Intercept term; the
            # drop is a no-op in that case.
            pass

        return params.dropna()

    def get_factor_names(self, level: int = 1) -> list[str]:
        """
        Get the factors in a model which correspond to a certain level.

        1 : pure factors
        2 : 2-factor interactions and quadratic terms
        3 : 3-factor interactions and cubic terms
        4 : etc
        """
        spec = ModelDesc.from_formula(self._model_spec)
        return [term.name() for term in spec.rhs_termlist if len(term.factors) == level]

    def get_response_name(self) -> str:
        """Get the name of the response variable from the model specification."""
        spec = ModelDesc.from_formula(self._model_spec)
        return spec.lhs_termlist[0].name()

    def get_title(self) -> str:
        """Get the model's title, if it has one. Always returns a string.

        The title is the data's ``pi_title``. ``lm`` also accepts a plain DataFrame,
        which has none, so a missing attribute is an empty title rather than an error.
        """
        return str(getattr(self.data, "pi_title", None) or "")

    def get_aliases(
        self,
        aliasing_up_to_level: int = 2,
        drop_intercept: bool | None = True,
        websafe: bool | None = False,
    ) -> list:
        """
        Return a list, containing strings, representing the aliases
        of the fitted effects.

        aliasing_up_to_level: up to which level of interactions shown

        drop_intercept: default is True, but sometimes it is interesting to
            know which effects are aliased with the intercept

        websafe: default is False; if True, will print the first term
            in the aliasing in bold, since that is the nominally estimated
            effect.
        """
        alias_strings: list[Any] = []
        if self.aliasing is None or len(self.aliasing.keys()) == 0:
            return alias_strings

        params = self.get_parameters(drop_intercept=bool(drop_intercept))
        for p_name in params.index.values:
            aliasing = f'<span style="font-size: 130%; font-weight: 700">{p_name}</span>' if websafe else p_name
            suffix = ""
            # Stored under the term's factors, ("A", "B") for "A:B"; ``get`` rather than
            # indexing, which on the defaultdict would add an empty entry per lookup.
            for alias in self.aliasing.get(tuple(p_name.split(":")), []):
                # Subtract "-1" because the first list entry tracks the sign
                if (len(alias) - 1) <= aliasing_up_to_level:
                    aliasing += f" {alias[0]} {':'.join(alias[1:])}"
                if (len(alias) - 1) > aliasing_up_to_level:
                    suffix = r" + <i>higher interactions</i>" if websafe else " + higher interactions"

            # Finished with this parameter
            alias_strings.append(aliasing + suffix)

        # All done
        return alias_strings


# Model.__repr__ = Model.__str__


def predict(model: Model, **kwargs: Any) -> Any:  # noqa: ANN401
    """Make predictions from the model."""
    return model._OLS.predict(exog={model._stand_ins.get(name, name): value for name, value in kwargs.items()})


#: Correlation beyond which two model columns are exact aliases: one, up to rounding.
_EXACT_ALIAS = 1.0 - 1e-9


def _alias_word(term: Any) -> list[str]:  # noqa: ANN401
    """Return the factor names of a patsy term, ``["Intercept"]`` for the intercept."""
    return [factor.name() for factor in term.factors] or ["Intercept"]


def _find_aliases(exog: np.ndarray, terms: list[Any], threshold: float) -> tuple[dict, list[int]]:
    """Find the model columns that are aliases of one another, and the ones to drop.

    Two columns that vary are aliases when the absolute correlation of the centred
    columns exceeds ``threshold`` (a threshold of one means exactly, up to rounding);
    the sign of an alias is the sign of that correlation, so it holds in actual units
    too (``E = 30 - A`` is ``-A``). Columns that do not vary (the intercept, or a word
    that is constant across a fraction, ``ABC = -1``) are aliases of one another,
    signed by their levels. In each group the shortest word is kept, the first one on a
    tie, and the others are dropped. ``terms`` holds each column's patsy term.

    Returns ``{word: [[sign, *alias word], ...]}``, each word a tuple of factor names
    with its aliases shortest first and the intercept last, and the columns to drop.
    """
    exog = np.asarray(exog, dtype=float)
    centred = exog - exog.mean(axis=0)
    spread = np.linalg.norm(centred, axis=0)
    varies = spread > np.sqrt(np.finfo(float).eps) * np.maximum(np.linalg.norm(exog, axis=0), 1.0)
    unit = np.divide(centred, spread, out=np.zeros_like(centred), where=varies)
    correlation = unit.T @ unit
    limit = min(threshold, _EXACT_ALIAS)

    aliasing: dict = defaultdict(list)
    drop: set[int] = set()
    for idx in range(exog.shape[1]):
        if varies[idx]:
            group = np.flatnonzero(varies & (np.abs(correlation[idx]) > limit))
            signs = np.sign(correlation[idx])
        else:
            group = np.flatnonzero(~varies)
            signs = np.sign(exog[0, idx] * exog[0])
        ordered = sorted(group.tolist(), key=lambda col: (len(terms[col].factors), col))
        drop.update(ordered[1:])
        aliases = [["-" if signs[col] < 0 else "+", *_alias_word(terms[col])] for col in ordered if col != idx]
        if aliases:
            aliasing[tuple(_alias_word(terms[idx]))] = sorted(
                aliases, key=lambda alias: (1e5 if alias[1] == "Intercept" else len(alias), alias)
            )
    return aliasing, sorted(drop)


def lm(
    model_spec: str,
    data: Expt,
    name: str | None = None,
    alias_threshold: float | None = _EXACT_ALIAS,
) -> Model:
    """Create a linear model.

    Parameters
    ----------
    model_spec : str
        The model formula, e.g. ``"y ~ A * B"``. A column named like a function the formula
        calls, as ``I`` in ``I(P ** 2)``, is fitted under a stand-in name and reported under its own.
    data : Expt
        The data; a plain ``Expt`` or DataFrame works too.
    name : str or None
        The model's name; defaults to the data's ``pi_title``, if it has one.
    alias_threshold : float or None
        Two model columns whose absolute correlation exceeds this are treated as
        aliases, and the longer term is dropped from the fit and reported as an alias.
        The default detects only exact aliasing (a correlation of one up to rounding),
        as in a fractional factorial; a column that is merely highly correlated, such
        as a square in actual units, is kept. ``None`` drops nothing.

    Returns
    -------
    Model
        The fitted model.
    """

    # Patsy evaluates each formula term as a Python expression, so an untrusted
    # ``model_spec`` is a code-execution vector. Allow only a safe Wilkinson
    # formula over the data columns, optionally with I()/Q() and a curated set of
    # element-wise numpy transforms (the public textbook API relies on these).
    stand_ins = stand_ins_for(model_spec, data.columns)
    fit_spec = with_stand_ins(model_spec, stand_ins)
    fit_data = data.rename(columns=stand_ins) if stand_ins else data
    validate_formula_is_safe(fit_spec, fit_data.columns, allow_transforms=True, allow_numpy=True)
    validate_contrast_levels(model_spec, data)

    pre_model = smf.ols(fit_spec, data=fit_data)
    # SEC-19 (#268): a formula like ``y ~ (A+B+C+D+E)**5`` expands to
    # 2**5 terms; combined with a wide ``data`` this is a CPU sink.
    # Cap the expanded term count after patsy parses the RHS.
    from process_improve.config import settings  # noqa: PLC0415

    n_terms = len(pre_model.data.xnames)
    if n_terms > settings.max_formula_terms:
        raise ValueError(
            f"formula {model_spec!r} expanded to {n_terms} terms; "
            f"the SEC-19 cap is settings.max_formula_terms="
            f"{settings.max_formula_terms}."
        )
    if alias_threshold is None:
        aliasing: dict = {}
        drop_column_names: list[str] = []
    else:
        # Map each model column to its patsy term: a categorical term spans several
        # columns, and patsy orders the terms in the model matrix its own way.
        spec: Any = getattr(pre_model.data, "model_spec", None) or pre_model.data.design_info
        column_terms: list[Any] = [None] * n_terms
        for term, columns in spec.term_slices.items():
            column_terms[columns] = [term] * (columns.stop - columns.start)
        aliasing, drop_columns = _find_aliases(pre_model.exog, column_terms, alias_threshold)
        drop_column_names = [pre_model.data.xnames[i] for i in drop_columns]

    post_model = smf.ols(fit_spec, data=fit_data, drop_cols=drop_column_names)
    if stand_ins:  # the coefficients, the summary and the aliases are read under the real names
        post_model.data.xnames = real_names(post_model.data.xnames, stand_ins)
        post_model.data.ynames = real_names(post_model.data.ynames, stand_ins)
        aliasing = real_names(aliasing, stand_ins)

    name = name or getattr(data, "pi_title", None)
    out = Model(
        OLS_instance=post_model.fit(),
        model_spec=model_spec,
        aliasing=aliasing,
        name=name,
    )
    out.data = data
    out._stand_ins = stand_ins

    return out


def summary(
    model: Model,
    show: bool | None = True,
    aliasing_up_to_level: int = 3,
) -> Any:  # noqa: ANN401
    """
    Print a summary to the screen of the model.

    Appends, if there is any aliasing, a summary of those aliases,
    up to the (integer) level of interaction: `aliasing_up_to_level`.
    """
    out = model.summary()
    extra = []
    aliases = model.get_aliases(aliasing_up_to_level, drop_intercept=False)
    values = model.get_parameters(drop_intercept=False).values
    if len(aliases):
        extra.append("Aliasing pattern")
        for value, alias in zip(values, aliases, strict=False):
            extra.append(f" {forg(float(value), 4)} = {alias}")

    out.add_extra_txt(extra)
    if show:
        print(out)  # noqa: T201
    return out


# ENG-23 (#305): explicit ``__all__`` so the thin re-exporter ``models.py``
# can do ``from ._lm import *`` without triggering CodeQL's
# py/polluting-import warning.
__all__ = [
    "Model",
    "UnsafeFormulaError",
    "forg",
    "lm",
    "predict",
    "suggest_identifier",
    "summary",
    "validate_formula_is_safe",
    "validate_identifier_is_safe",
    "validate_identifiers_are_safe",
]
