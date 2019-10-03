import numpy as np
import pandas as pd

from structures import (gather, c, expand_grid)
from models import (lm, summary)
from plotting import (pareto_plot, contour_plot)


def case_3B():
    """
    See video 3B in the Coursera series. R code equivalent: http://yint.org/3B

    Two factors, no extra degrees of freedom.
    """
    A = c(-1, +1, -1, +1, name='Additive')
    B = c(-1, -1, +1, +1, name='Blender')
    y = c(52, 74, 62, 80, units='number of popped kernels')

    expt = gather(A=A, B=B, y=y, title='Popping corn!')
    popped_corn = lm("y ~ A + B + A*B", expt)
    popped_corn = lm("y ~ A*B", expt)
    summary(popped_corn)
    contour_plot(popped_corn, show=False)


def case_3C():
    """
    See video 3C in the Coursera series. R code equivalent: http://yint.org/3C

    3 factors, no extra degrees of freedom.
    """
    C = T = S = c(-1, +1)
    C, T, S = expand_grid(C=C, T=T, S=S)
    y = c(5, 30, 6, 33, 4, 3, 5, 4)
    expt = gather(C=C, T=T, S=S, y=y)

    water = lm("y ~ C * T * S", expt)
    summary(water)
    contour_plot(water, "C", "T")
    pareto_plot(water)

def case_3D():
    """
    solar = data("solar")
    model_y1 = lm("y1 ~ A*B*C*D", data=solar)
    summary(model_y1)
    pareto_plot(model_y1)

    model_y2 = lm("y2 ~ A*B*C*D", data=solar)
    summary(model_y2)
    pareto_plot(model_y2)
    pareto_plot(model_y2, main="Pareto plot for Energy Delivery Efficiency")

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
    pareto_plot(mod_ff)

    # Now rebuild the linear model with only the 4 important terms
    mod_res4 = lm("y ~ A*C*E*G", expt)
    pareto_plot(mod_res4)
    """
    pass

def case_w2():
    """
    Teaching case week 2: https://yint.org/w2
    """
    # T = time used for baking:
    #      (-1) corresponds to 80 minutes and (+1) corresponds to 100 minutes
    T = c(-1, +1, -1, +1, lo=80, hi=100)

    # F = quantity of fat used:
    #      (-1) corresponds to 20 g and (+1) corresponds to 30 grams
    F = c(-1, -1, +1, +1, lo=20, hi=30)

    # Response y is the crispiness
    y = c(37, 57, 49, 53, units='crispiness')

    # Fit a linear model
    expt = gather(T=T, F=F, y=y)
    model_crispy = lm("y ~ T + F + T*F", expt)
    summary(model_crispy)

    # See how the two factors affect the response:
    contour_plot(model_crispy )
    interaction_plot(T, F, y)
    interaction_plot(F, T, y)

    # Make a prediction with this model:
    xT = +2   # corresponds to 110 minutes
    xF = -1   # corresponds to 20 grams of fat
    y.hat = predict(model_crispy, data.frame(T = xT, F = xF))
    paste0('Predicted value is: ', y.hat, ' crispiness.')


def case_w4_1():
    """
    Teaching case week 4: https://yint.org/w4
    """
    # S = Free shipping if order amount is €30 or more [-1],
    # or if order amount is over €50 [+1]
    S = c(-1, +1, -1, +1, -1, +1, -1, +1, name='Free shipping amount')

    # Does the purchaser need to create a profile first [+1] or not [-1]?
    P = c(-1, -1, +1, +1, -1, -1, +1, +1, name='Create profile: No/Yes')

    # Response: daily sales amount
    y = c(348, 359, 327, 243, 356, 363, 296, 257)

    # Linear model using S, P and S*P to predict the response
    expt = gather(S=S, P=P, y=y, title='Experiment without mistake')
    model_sales = lm("y ~ S*P", expt)
    summary(model_sales)
    contour_plot(model_sales)


def case_w4_2():
    """
    Teaching case week 4: https://yint.org/w4
    """
    # S = Free shipping if order amount is €30 or more [-1], or if
    # order amount is over €50 [+1]. Notice that a mistake was made
    # with the last experiment: order minimum for free shipping was €60 [+1].
    S = c(-1, +1, -1, +1, -1, +1, -1, +2, name='Free shipping amount')

    # Does the purchaser need to create a profile first [+1] or not [-1]?
    P = c(-1, -1, +1, +1, -1, -1, +1, +1, name='Create profile: No/Yes')

    # Response: daily sales amount
    y = c(348, 359, 327, 243, 356, 363, 296, 220, units='€ sales')

    # Linear model using S, P and S*P to predict the response
    expt = gather(S=S, P=P, y=y, title='Experiment with mistake')
    model_sales_mistake = lm("y ~ S*P", expt)
    summary(model_sales_mistake)

    contour_plot(model_sales_mistake)


if __name__ == '__main__':
    #case_3B()
    #case_3C()
    #case_3D()

    #case_w2()
    case_w4_1()
    case_w4_2()


