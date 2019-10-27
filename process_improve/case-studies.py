import numpy as np
import pandas as pd

import os
import sys
sys.path.append(os.path.split(__file__)[0])

from structures import (gather, c, expand_grid, supplement)
from models import (lm, summary, predict)
from plotting import (pareto_plot, contour_plot, tradeoff_table)
from designs_factorial import full_factorial

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

def case_3C(show=False):
    """
    See video 3C in the Coursera series. R code equivalent: http://yint.org/3C

    3 factors, no extra degrees of freedom.
    """
    C = T = S = c(-1, +1)
    C, T, S = expand_grid(C=C, T=T, S=S)
    y = c(5, 30, 6, 33, 4, 3, 5, 4)
    expt = gather(C=C, T=T, S=S, y=y, title='Water treatment')

    water = lm("y ~ C * T * S", expt)
    summary(water)
    if show:
        contour_plot(water, "C", "T", show=show)
        pareto_plot(water, show=show, up_to_level=2)

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
    #interaction_plot(T, F, y)
    #interaction_plot(F, T, y)

    # Make a prediction with this model:
    xT = +2   # corresponds to 110 minutes
    xF = -1   # corresponds to 20 grams of fat
    #y.hat = predict(model_crispy, pd.DataFrame(T = xT, F = xF))
    #paste0('Predicted value is: ', y.hat, ' crispiness.')

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

def case_worksheet_5():
    """
    We have a bioreactor system, and we are investigating four factors:
    A = feed rate 				          5 g/min or   8 g/min
    B = initial inoculant amount 		300 g     or 400 g
    C = feed substrate concentration 	 40 g/L   or  60 g/L
    D = dissolved oxygen set-point 	      4 mg/L  or   5 mg/L

    The 16 experiments from a full factorial, were randomly run, and the yields
    from the bioreactor, y, are reported here in standard order:
    y = [60, 59, 63, 61, 69, 61, 94, 93, 56, 63, 70, 65, 44, 45, 78, 77].
    The yield has units of g/L.

    Without running any code or calculations, answer these questions:
    * how many 2-factor interactions are there in this full factorial: _______
    * how many 3-factor interactions are there in this full factorial: _______
    * how many 4-factor interactions are there in this full factorial: _______
    * how many terms can you fit in a full linear model, including the intercept
    * how many data points do you have to fit this model: _________
    * what will be the value of R2 if you fit this full linear model: _______
    * and the standard error will be exactly: ________


    Run your adjusted code and check that the values of A, B, C and D in the
    vector are in the order they should be.

    Use a Pareto-plot to identify the significant effects.

    Rebuild the model now without the factor, or factors which have the least
    influence on the model. Compare the existing model with this newly updated
    model. What do you notice about the coefficients?


    What would be your advice to your colleagues to improve the yield?

    Which main effects should you change?

    Will the interaction(s) of this main effect, or these main effects,
    work in your favour, or work against you?

    Make predictions of experiments in several directions of the main factors,
    to try to maximize the reactor yield.
    """

    A, B, C, D = full_factorial(4, names=['A', 'B', 'C', 'D'])

    A = supplement(A, name = 'Feed rate', units='g/min', lo = 5, high = 8.0)
    B = supplement(B, name = 'Initial inoculant amount', units = 'g', lo = 300,
                   hi = 400)
    C = supplement(C, name = 'Feed substrate concentration', units = 'g/L',
                   lo = 40, hi = 60)
    D = supplement(D, name = 'Dissolved oxygen set-point', units = 'mg/L',
                   lo = 4, hi = 5)

    y = c(60, 59, 63, 61, 69, 61, 94, 93, 56, 63, 70, 65, 44, 45, 78, 77,
          units='g/L', name = 'Yield')

    expt = gather(A=A, B=B, C=C, D=D, y=y,
                  title='Initial experiments; full factorial')
    model_start = lm("y ~ A*B*C*D", expt)

    summary(model_start)
    #pareto_plot(model_start, plot_width=800)
    contour_plot(model_start, "A", "B")
    contour_plot(model_start, "B", "C")
    contour_plot(model_start, "C", "D")

def api_usage():

    A, B, C, D = full_factorial(4, names = ['A', 'B', 'C', 'D'])

    A = supplement(A, name = 'Feed rate', units='g/min', lo = 5, high = 8.0)
    B = supplement(B, name = 'Initial inoculant amount', units = 'g', lo = 300,
                   hi = 400)
    C = supplement(C, name = 'Feed substrate concentration', units = 'g/L',
                   lo = 40, hi = 60)
    D = supplement(D, name = 'Dissolved oxygen set-point', units = 'mg/L',
                   lo = 4, hi = 5)

    expt = gather(A, B, C, D, title='Initial experiments; full factorial')

    expt.show_actual(std_order=True)
    expt.show_actual(random_order=True, seed=13)
    expt.show_coded()
    expt.power()
    expt.export(save_as='xls' , filename='abc.xlsx')


    center_points = expt.get_center_points()
    expt.append(center_points)
    cp_expt = 70
    expt['y'] = c(60, 59, 63, 61, 69, 61, 94, 93, 56, 63, 70, 65, 44, 45, 78,
                  77, cp_expt, units='g/L', name = 'Yield')


    model_start = lm("y ~ <full>", expt)
    model_start = lm("y ~ <2fi>", expt)
    model_start = lm("y ~ <3fi>", expt)

def case_worksheet_6():
    """
    Half-fraction
    """
    # The half-fraction, when C = A*B
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    C = A * B

    # The response: stability [units=days]
    y = c(41, 27, 35, 20, name="Stability", units="days")

    # Linear model using only 4 experiments
    expt = gather(A=A, B=B, C=C, y=y, title='Half-fraction, using C = A*B')
    print(expt)
    model_stability_poshalf = lm("y ~ A*B*C", expt)
    summary(model_stability_poshalf)
    pareto_plot(model_stability_poshalf)
    pareto_plot(model_stability_poshalf, up_to_level=2)

if __name__ == '__main__':
# tradeoff_table()
    #case_3B()
    # case_3C(show=True)
    #case_3D()
    # case_worksheet_5()
    # api_usage()
    case_worksheet_6()

    case_w2()
    case_w4_1()
    case_w4_2()


