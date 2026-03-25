# (c) Kevin Dunn, 2010-2026. MIT License. Based on own private work over the years.

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import ModelDesc
from statsmodels.regression.linear_model import OLS


def forg(x: float, prec: int = 3) -> str:
    """Yanked from the code for Statsmodels / iolib / summary.py and adjusted."""
    if prec == 3:
        # for 3 decimals
        if (abs(x) >= 1e4) or (abs(x) < 1e-4):
            return f"{x:9.3g}"
        else:
            return f"{x:9.3f}"
    elif prec == 4:
        if (abs(x) >= 1e4) or (abs(x) < 1e-4):
            return f"{x:10.4g}"
        else:
            return f"{x:10.4f}"
    else:
        raise NotImplementedError


class Model(OLS):
    """Just a thin wrapper around the OLS class from Statsmodels."""

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

    def __str__(self) -> str:
        """Return the model specification as a string."""
        spec = ModelDesc.from_formula(self._model_spec)
        return spec.describe()

    def summary(self, alpha: float = 0.05, print_to_screen: bool = True) -> Any:  # noqa: ARG002, ANN401
        """Side effect: prints to the screen."""
        # Taken from statsmodels.regression.linear_model.py
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            main = "OLS Regression Results"
            if self.name:
                main += ": " + str(self.name)
            elif self.data.pi_title:
                main += ": " + str(self.data.pi_title)

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

        return smry

    def get_parameters(self, drop_intercept: bool = True) -> pd.DataFrame:
        """Get the parameter values; return them in a Pandas dataframe."""

        params = self._OLS.params.copy()
        try:
            if drop_intercept:
                params = params.drop("Intercept")
        except KeyError:
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
        """Get the model's title, if it has one. Always returns a string."""
        return self.data.get_title()

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
        if len(self.aliasing.keys()) == 0:
            return alias_strings

        params = self.get_parameters(drop_intercept=drop_intercept)
        for p_name in params.index.values:
            aliasing = f'<span style="font-size: 130%; font-weight: 700">{p_name}</span>' if websafe else p_name
            suffix = ""
            for alias in self.aliasing[(p_name,)]:
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
    return model._OLS.predict(exog=dict(kwargs))


def lm(  # noqa: C901, PLR0915
    model_spec: str,
    data: pd.DataFrame,
    name: str | None = None,
    alias_threshold: float | None = 0.995,
) -> Model:
    """Create a linear model."""

    def find_aliases(  # noqa: C901, PLR0912
        model: Any,  # noqa: ANN401
        model_desc: ModelDesc,
        threshold_correlation: float = 0.995,
    ) -> tuple[dict, list]:
        """
        Find columns which are exactly correlated, or up to at least a level
        of `threshold_correlation`.
        Return a dictionary of aliasing and a list of columns to keep.

        The columns to keep will be in the order checked. Perhaps this can be
        improved.
        For example if AB = CD, then return AB to keep.
        For example if A = BCD, then return A, and not the BCD column to keep.
        """
        has_variation = model.exog.std(axis=0) > np.sqrt(np.finfo(float).eps)

        # np.dot(model.exog.T, model.exog)/model.exog.shape[0]
        # Drop columns which do not have any variation
        corrcoef = np.corrcoef(model.exog[:, has_variation].T)  # , ddof=0)

        # Snippet of code here is from the NumPy "corrcoef" function. Adapted.
        c = np.cov(model.exog.T, None, rowvar=True)
        dot_product = model.exog.T @ model.exog
        try:
            d = np.diag(c)
        except ValueError:
            # scalar covariance
            # nan if incorrect value (nan, inf, 0), 1 otherwise
            return c / c
        stddev = np.sqrt(d.real)

        aliasing = defaultdict(list)
        terms = model_desc.rhs_termlist
        drop_columns = []
        counter = -1
        corrcoef = c.copy()
        for idx, check in enumerate(has_variation):
            if check:
                counter += 1

                for j, stddev_value in enumerate(stddev):
                    if stddev_value == 0:
                        pass
                    else:
                        corrcoef[idx, j] = c[idx, j] / stddev[idx] / stddev_value

                # corrcoef = c / stddev[idx, None]
                # corrcoef = corrcoef / stddev[None, idx]

                candidates = [i for i, val in enumerate(np.abs(corrcoef[idx, :])) if (val > threshold_correlation)]
                signs = [np.sign(j) for j in corrcoef[idx, :]]
            else:
                # Columns with no variation
                candidates = [i for i, j in enumerate(has_variation) if (j <= threshold_correlation)]

            # Track the correlation signs
            signs = [np.sign(j) for j in dot_product[idx, :]]

            # Now drop out the candidates with the longest word lengths
            alias_len = [(len(terms[i].factors), i) for i in candidates]
            alias_len.sort(reverse=True)
            drop_columns.extend(entry[1] for entry in alias_len[0:-1])

            for col in candidates:
                if col == idx:
                    # It is of course perfectly correlated with itself
                    pass
                else:
                    aliases = [t.name() for t in terms[col].factors]
                    if len(aliases) == 0:
                        aliases = ["Intercept"]

                    key = tuple([t.name() for t in terms[idx].factors])
                    if len(key) == 0:
                        key = ("Intercept",)

                    if signs[col] > 0:
                        aliases.insert(0, "+")
                    if signs[col] < 0:
                        aliases.insert(0, "-")
                    aliasing[key].append(aliases)

        # Sort the aliases in length:
        for key, val in aliasing.items():
            alias_len = [(len(i), i) if i[1] != "Intercept" else (1e5, i) for i in val]
            alias_len.sort()
            aliasing[key] = [i[1] for i in alias_len]

        return aliasing, list(set(drop_columns))

    pre_model = smf.ols(model_spec, data=data)
    model_description = ModelDesc.from_formula(model_spec)
    aliasing, drop_columns = find_aliases(pre_model, model_description, threshold_correlation=alias_threshold)
    drop_column_names = [pre_model.data.xnames[i] for i in drop_columns]

    post_model = smf.ols(model_spec, data=data, drop_cols=drop_column_names)

    name = name or data.pi_title
    out = Model(
        OLS_instance=post_model.fit(),
        model_spec=model_spec,
        aliasing=aliasing,
        name=name,
    )
    out.data = data

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
            extra.append(f" {forg(value, 4)} = {alias}")

    out.add_extra_txt(extra)
    if show:
        print(out)  # noqa: T201
    return out
