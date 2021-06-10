# (c) Kevin Dunn, 2010-2021. MIT License. Based on own private work over the years.

import warnings
from collections import defaultdict
from typing import Optional, List, Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import OLS
from patsy import ModelDesc


def forg(x, prec=3):
    """
    Yanked from the code for Statsmodels / iolib / summary.py and adjusted.
    """
    if prec == 3:
        # for 3 decimals
        if (abs(x) >= 1e4) or (abs(x) < 1e-4):
            return "%9.3g" % x
        else:
            return "%9.3f" % x
    elif prec == 4:
        if (abs(x) >= 1e4) or (abs(x) < 1e-4):
            return "%10.4g" % x
        else:
            return "%10.4f" % x
    else:
        raise NotImplementedError


class Model(OLS):
    """
    Just a thin wrapper around the OLS class from Statsmodels."""

    def __init__(self, OLS_instance, model_spec, aliasing=None, name=None):
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

    def __str__(self):
        spec = ModelDesc.from_formula(self._model_spec)
        return spec.describe()

    def summary(self, alpha=0.05, print_to_screen=True):
        """
        Side effect: prints to the screen.
        """
        # Taken from statsmodels.regression.linear_model.py
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            main = "OLS Regression Results"
            if self.name:
                main += ": " + str(self.name)
            else:
                if self.data.pi_title:
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

    def get_parameters(self, drop_intercept=True) -> pd.DataFrame:
        """Gets the paramter values; returns it in a Pandas dataframe"""

        params = self._OLS.params.copy()
        try:
            if drop_intercept:
                params.drop("Intercept", inplace=True)
        except KeyError:
            pass

        params.dropna(inplace=True)
        return params

    def get_factor_names(self, level=1):
        """
        Gets the factors in a model which correspond to a certain level:
        1 : pure factors
        2 : 2-factor interactions and quadratic terms
        3 : 3-factor interactions and cubic terms
        4 : etc
        """
        spec = ModelDesc.from_formula(self._model_spec)
        return [term.name() for term in spec.rhs_termlist if len(term.factors) == level]

    def get_response_name(self):
        spec = ModelDesc.from_formula(self._model_spec)
        return spec.lhs_termlist[0].name()

    def get_title(self) -> str:
        """Gets the model's title, if it has one. Always returns a string."""
        return self.data.get_title()

    def get_aliases(
        self,
        aliasing_up_to_level: int = 2,
        drop_intercept: Optional[bool] = True,
        websafe: Optional[bool] = False,
    ) -> list:
        """
        Returns a list, containing strings, representing the aliases
        of the fitted effects.

        aliasing_up_to_level: up to which level of interactions shown

        drop_intercept: default is True, but sometimes it is interesting to
            know which effects are aliased with the intercept

        websafe: default is False; if True, will print the first term
            in the aliasing in bold, since that is the nominally estimated
            effect.
        """
        alias_strings: List[Any] = []
        if len(self.aliasing.keys()) == 0:
            return alias_strings

        params = self.get_parameters(drop_intercept=drop_intercept)
        for p_name in params.index.values:
            if websafe:
                aliasing = (
                    '<span style="font-size: 130%; font-weight: 700">'
                    f"{p_name}</span>"
                )
            else:
                aliasing = p_name
            suffix = ""
            for alias in self.aliasing[tuple([p_name])]:

                # Subtract "-1" because the first list entry tracks the sign
                if (len(alias) - 1) <= aliasing_up_to_level:
                    aliasing += f" {alias[0]} {':'.join(alias[1:])}"
                if (len(alias) - 1) > aliasing_up_to_level:
                    if websafe:
                        suffix = r" + <i>higher interactions</i>"
                    else:
                        suffix = " + higher interactions"

            # Finished with this parameter
            alias_strings.append(aliasing + suffix)

        # All done
        return alias_strings


# Model.__repr__ = Model.__str__


def predict(model, **kwargs):
    """
    Make predictions from the model
    """
    return model._OLS.predict(exog=dict(kwargs))


def lm(
    model_spec: str,
    data: pd.DataFrame,
    name: Optional[str] = None,
    alias_threshold: Optional[float] = 0.995,
) -> Model:
    """
    Create a linear model.
    """

    def find_aliases(model, model_desc, threshold_correlation=0.995):
        """
        Finds columns which are exactly correlated, or up to at least a level
        of `threshold_correlation`.
        Returns a dictionary of aliasing and a list of columns to keep.

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
                        corrcoef[idx, j] = c[idx, j] / stddev[idx] / stddev[j]

                # corrcoef = c / stddev[idx, None]
                # corrcoef = corrcoef / stddev[None, idx]

                candidates = [
                    i
                    for i, val in enumerate(np.abs(corrcoef[idx, :]))
                    if (val > threshold_correlation)
                ]
                signs = [np.sign(j) for j in corrcoef[idx, :]]
            else:
                # Columns with no variation
                candidates = [
                    i
                    for i, j in enumerate(has_variation)
                    if (j <= threshold_correlation)
                ]

            # Track the correlation signs
            signs = [np.sign(j) for j in dot_product[idx, :]]

            # Now drop out the candidates with the longest word lengths
            alias_len = [(len(terms[i].factors), i) for i in candidates]
            alias_len.sort(reverse=True)
            for entry in alias_len[0:-1]:
                drop_columns.append(entry[1])

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
    aliasing, drop_columns = find_aliases(
        pre_model, model_description, threshold_correlation=alias_threshold
    )
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
    show: Optional[bool] = True,
    aliasing_up_to_level: int = 3,
):
    """
    Prints a summary to the screen of the model.

    Appends, if there is any aliasing, a summary of those aliases,
    up to the (integer) level of interaction: `aliasing_up_to_level`.
    """
    out = model.summary()
    extra = []
    aliases = model.get_aliases(aliasing_up_to_level, drop_intercept=False)
    values = model.get_parameters(drop_intercept=False).values
    if len(aliases):
        extra.append("Aliasing pattern")
        for value, alias in zip(values, aliases):
            extra.append(f" {forg(value, 4)} = {alias}")

    out.add_extra_txt(extra)
    if show:
        print(out)
    return out
