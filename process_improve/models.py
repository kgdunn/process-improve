# (c) Kevin Dunn, 2019. MIT License.

from typing import Optional
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from patsy import ModelDesc

class Model(OLS):
    """
    Just a thin wrapper around the OLS class from Statsmodels."""
    def __init__(self, OLS_instance, model_spec, aliasing=None):
        self._OLS = OLS_instance
        self._model_spec = model_spec

        # Standard error
        self.df_resid = self._OLS.df_resid
        self.df_model = self._OLS.df_model
        self.nobs = self._OLS.nobs
        # Leads to errors for size inconsistency if the data frames have
        # missing data?
        #self.rsquared = self.R2 = self._OLS.rsquared
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

            main = 'OLS Regression Results'
            if self.data.pi_title:
                main += ': ' + str(self.data.pi_title)

            smry = self._OLS.summary(title=main)
            # print(smry)
            # Call this directly and modify the result to suppress what we
            # don't really care to show:

            smry.tables[0].pop(8)

            se = '---'
            if not(np.isinf(self._OLS.scale)):
                se = f'{np.sqrt(self._OLS.scale):.3f}'

            #Residual standard error
            smry.tables[0][7][0].data = 'Residual std error'
            smry.tables[0][7][1].data = se
            #smry.tables[0][7][0].data = se
            #smry.tables[0][7][1].data = se

        return smry

    def get_parameters(self, drop_intercept=True) -> pd.DataFrame:
        """Gets the paramter values; returns it in a Pandas dataframe"""

        params = self._OLS.params.copy()
        try:
            if drop_intercept:
                params.drop('Intercept', inplace=True)
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
        return [term.name() for term in spec.rhs_termlist \
                                                if len(term.factors)==level]


    def get_title(self) -> str:
        """ Gets the model's title, if it has one. Always returns a string."""
        return self.data.get_title()



Model.__repr__ = Model.__str__

def predict(model, **kwargs):
    """
    Make predictions from the model
    """
    # kwargs
    return model._OLS.predict(exog=dict(kwargs))



def lm(model_spec: str, data: pd.DataFrame) -> Model:
    """
    Create a linear model.
    """
    # TODO: handle collinear columns, aliases.
    #

    def find_aliases(model, model_desc):
        """
        Finds columns which are exactly correlated.
        Returns a dictionary of aliasing and a list of columns to keep.

        The columns to keep will be in the order checked. Perhaps this can be
        improved.
        For example if AB = CD, then return AB to keep.
        For example if A = BCD, then return A, and not the BCD column to keep.
        """
        has_variation = model.exog.std(axis=0) > np.sqrt(np.finfo(float).eps)

        #np.dot(model.exog.T, model.exog)/model.exog.shape[0]
        # Drop columns whihc do not have any variation
        corrcoef = np.corrcoef(model.exog[:, has_variation].T) #, ddof=0)
        aliasing = defaultdict(list)
        lim = 0.9995
        terms = model_desc.rhs_termlist

        drop_columns = []
        keep_columns = list(range(len(has_variation)))
        counter = -1
        for idx, check in enumerate(has_variation):#enumerate(range(cc.shape[1])):
            if check:
                counter += 1

            candidates = [i for i,j in enumerate(np.abs(corrcoef[counter])) if (j>lim)]
            alias_len = [(i, len(terms[i].factors)) for i in candidates]
            alias_len.sort(reverse=True)
            for entry in alias_len[0:-1]:
                drop_columns.append(entry[0])
                try:
                    keep_columns.pop(keep_columns.index(entry[0]))
                except ValueError:
                    pass

            for col in candidates:
                if col == idx:
                    # It is of course perfectly correlated with itself
                    pass
                else:
                    model_desc.rhs_termlist[col].factors
                    aliasing[terms[idx].factors].append(terms[col].factors)


        return aliasing, list(set(drop_columns))


    pre_model = smf.ols(model_spec, data=data)
    model_description = ModelDesc.from_formula(model_spec)
    aliasing, drop_columns = find_aliases(pre_model, model_description)
    drop_column_names = [pre_model.data.xnames[i] for i in drop_columns]

    post_model = smf.ols(model_spec, data=data, drop_cols=drop_column_names)
    out = Model(OLS_instance=post_model.fit(),
                model_spec=model_spec,
                aliasing=aliasing)
    out.data = data

    return out


def summary(model: Model, show: Optional[bool] = True):
    """
    Prints a summary to the screen of the model.
    """
    out = model.summary()
    if show:
        print(out)
    return out

