# (c) Kevin Dunn, 2019.

#import statsmodels.api as sm
#import statsmodels.formula.api as smf
import pandas as pd


def c(*args, **kwargs) -> list:
    """
    Performs the equivalent of the R function "c(...)", to combine data elements
    into a list. Converts every entry into a floating point object.
    """
    return [float(j) for j in args]


def gather(**kwargs):
    """
    Gathers the named inputs together as columns for a data frame.

    """
    out = pd.DataFrame(data=None, index=None, columns=None, dtype=None  )
    lens = [len(value) for value in kwargs.values()]
    avg_count = pd.np.median(lens)
    for key, value in kwargs.items():
        assert len(value) == avg_count, f"Column {key} must have length {avg_count}."
        out[key] = value

    return out

def lm(model_spec: str, df: pd.DataFrame):
    """
    """
    out = {}
    #results = smf.ols('y ~ A*B', data=dat).fit()
    return out

def summary(model: dict):
    """
    Prints a summary to the screen of the model.
    """
    return None


if __name__ == '__main__':
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    y = c(52, 74, 62, 80)

    expt = gather(A=A, B=B, y=y)
    popped_corn = lm("y ~ A + B + A*B", expt)
    summary(popped_corn)


