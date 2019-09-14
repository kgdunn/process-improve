# (c) Kevin Dunn, 2019. MIT License.

#import statsmodels.api as sm
#import statsmodels.formula.api as smf
import pandas as pd

from plotting import paretoPlot
from datasets import data
from structures import c

class expand(object):
    pass

    def grid(**kwargs):
        """
        Create the expanded grid here.
        """
        print('TODO')
        return kwargs.values()


def gather(*args, **kwargs):
    """
    Gathers the named inputs together as columns for a data frame.

    """
    # TODO : handle the case where the shape of an input >= 2 columns

    out = pd.DataFrame(data=None, index=None, columns=None, dtype=None)
    lens = [len(value) for value in kwargs.values()]
    avg_count = pd.np.median(lens)
    index = []
    for key, value in kwargs.items():
        #assert len(value) == avg_count, (f"Column {key} must have length "
        #                                 f"{avg_count}.")
        if isinstance(value, list):
            out[key] = value
        elif isinstance(value, pd.DataFrame):
            out[key] = value.values.ravel()

            if hasattr(value, '_pi_index'):
                index.append(value.index)

    # TODO : check that all indexes are common, to merge. Or use the pandas
    #        functionality of merging series with the same index

    return out


def lm(model_spec: str, data: pd.DataFrame):
    """
    """
    out = {}
    print('TODO')
    #results = smf.ols('y ~ A*B', data=dat).fit()
    return out


def summary(model: dict):
    """
    Prints a summary to the screen of the model.
    """
    print('TODO')
    return None


if __name__ == '__main__':

    # 3B
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    y = c(52, 74, 62, 80)

    expt = gather(A=A, B=B, y=y)
    popped_corn = lm("y ~ A + B + A*B", expt)
    summary(popped_corn)

    # 3C
    C = T = S = c(-1, +1)
    C, T, S = expand.grid(C=C, T=T, S=S)
    y = c(5, 30, 6, 33, 4, 3, 5, 4)
    expt = gather(C=C, T=T, S=S, y=y)

    water = lm("y ~ C * T * S", expt)
    summary(water)


    paretoPlot(water)

    #3D
    solar = data("solar")
    model_y1 = lm("y1 ~ A*B*C*D", data=solar)
    summary(model_y1)
    paretoPlot(model_y1)

    model_y2 = lm("y2 ~ A*B*C*D", data=solar)
    summary(model_y2)
    paretoPlot(model_y2)
    paretoPlot(model_y2, main="Pareto plot for Energy Delivery Efficiency")


    # 4H
    A = B = C = c(-1, +1)
    A, B, C = expand.grid(A=A, B=B, C=C)

    # These 4 factors are generated, using the trade-off table relationships
    D = A*B
    E = A*C
    F = B*C
    G = A*B*C

    # These are the 8 experimental outcomes, corresponding to the 8 entries
    # in each of the vectors above
    y = c(320, 276, 306, 290, 272, 274, 290, 255)

    expt = gather(A=A, B=B, C=C, D=D, E=E, F=F, G=G, y=y)

    # And finally, the linear model
    mod_ff = lm("y ~ A*B*C*D*E*F*G", expt)
    paretoPlot(mod_ff)

    # Now rebuild the linear model with only the 4 important terms
    mod_res4 = lm("y ~ A*C*E*G", expt)
    paretoPlot(mod_res4)




