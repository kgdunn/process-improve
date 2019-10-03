# (c) Kevin Dunn, 2019. MIT License.

from typing import Optional
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import OLS
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import Cell


class Model(OLS):
    """
    Just a thin wrapper around the OLS class from Statsmodels."""
    def __init__(self, OLS_instance):
        self._OLS = OLS_instance

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

            #Residual standard error
            se = '---'
            if not(np.isinf(self._OLS.scale)):
                se = f'{np.sqrt(self._OLS.scale):.3f}'
            smry.tables[0][7][0].data = 'Residual std error'
            smry.tables[0][7][1].data = se
            #smry.tables[0][7][0].data = se
            #smry.tables[0][7][1].data = se

        return smry


def predict(model, **kwargs):
    """
    Make predictions from the model
    """
    #kwargs
    return model._OLS.predict(exog=dict(kwargs))



def lm(model_spec: str, data: pd.DataFrame) -> Model:
    """
    """
    model = smf.ols(model_spec, data=data).fit()
    out = Model(OLS_instance=model)
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

