# (c) Kevin Dunn, 2019. MIT License.

from typing import Optional
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import OLS
from patsy import ModelDesc

class Model(OLS):
    """
    Just a thin wrapper around the OLS class from Statsmodels."""
    def __init__(self, OLS_instance, model_spec):
        self._OLS = OLS_instance
        self._model_spec = model_spec

        # Standard error
        self.df_resid = self._OLS.df_resid
        self.df_model = self._OLS.df_model
        self.nobs = self._OLS.nobs
        self.rsquared = self.R2 = self._OLS.rsquared
        self.residuals = self._OLS.resid

        # Will be replaced by the "lm()" function
        self.data = None

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
            if hasattr(self.data, '_pi_title'):
                main += ': ' + str(getattr(self.data, '_pi_title'))

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
        return getattr(self.data, '_pi_title', '')



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
    model = smf.ols(model_spec, data=data).fit()
    out = Model(OLS_instance=model, model_spec=model_spec)
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

