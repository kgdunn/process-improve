# (c) Kevin Dunn, 2010-2025. MIT License. Based on own private work over the years.

from .designs_factorial import full_factorial
from .models import lm, predict, summary
from .plotting import contour_plot, pareto_plot, plot_model
from .structures import c, expand_grid, gather, supplement


def case_3B():
    """
    See video 3B in the Coursera series. R code equivalent: http://yint.org/3B

    Two factors, no extra degrees of freedom.
    """
    A = c(-1, +1, -1, +1, name="Additive")
    B = c(-1, -1, +1, +1, name="Blender")
    y = c(52, 74, 62, 80, units="number of popped kernels")

    expt = gather(A=A, B=B, y=y, title="Popping corn!")
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
    expt = gather(C=C, T=T, S=S, y=y, title="Water treatment")

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


def case_w2():
    """Teaching case week 2: https://yint.org/w2"""
    # T = time used for baking:
    #      (-1) corresponds to 80 minutes and (+1) corresponds to 100 minutes
    T = c(-1, +1, -1, +1, lo=80, hi=100)

    # F = quantity of fat used:
    #      (-1) corresponds to 20 g and (+1) corresponds to 30 grams
    F = c(-1, -1, +1, +1, lo=20, hi=30)

    # Response y is the crispiness
    y = c(37, 57, 49, 53, units="crispiness")

    # Fit a linear model
    expt = gather(T=T, F=F, y=y)
    model_crispy = lm("y ~ T + F + T*F", expt)
    summary(model_crispy)

    # See how the two factors affect the response:
    contour_plot(model_crispy)
    # interaction_plot(T, F, y)
    # interaction_plot(F, T, y)

    # Make a prediction with this model:
    xT = +2  # corresponds to 110 minutes
    xF = -1  # corresponds to 20 grams of fat
    y_hat = predict(model_crispy, T=xT, F=xF)
    print(f"Predicted value is: {y_hat} crispiness.")


def case_w4_1():
    """Teaching case week 4: https://yint.org/w4"""
    # S = Free shipping if order amount is €30 or more [-1],
    # or if order amount is over €50 [+1]
    S = c(-1, +1, -1, +1, -1, +1, -1, +1, name="Free shipping amount")

    # Does the purchaser need to create a profile first [+1] or not [-1]?
    P = c(-1, -1, +1, +1, -1, -1, +1, +1, name="Create profile: No/Yes")

    # Response: daily sales amount
    y = c(348, 359, 327, 243, 356, 363, 296, 257)

    # Linear model using S, P and S*P to predict the response
    expt = gather(S=S, P=P, y=y, title="Experiment without mistake")
    model_sales = lm("y ~ S*P", expt)
    summary(model_sales)
    contour_plot(model_sales)


def case_w4_2():
    """Teaching case week 4: https://yint.org/w4"""
    # S = Free shipping if order amount is €30 or more [-1], or if
    # order amount is over €50 [+1]. Notice that a mistake was made
    # with the last experiment: order minimum for free shipping was €60 [+1].
    S = c(-1, +1, -1, +1, -1, +1, -1, +2, name="Free shipping amount")

    # Does the purchaser need to create a profile first [+1] or not [-1]?
    P = c(-1, -1, +1, +1, -1, -1, +1, +1, name="Create profile: No/Yes")

    # Response: daily sales amount
    y = c(348, 359, 327, 243, 356, 363, 296, 220, units="€ sales")

    # Linear model using S, P and S*P to predict the response
    expt = gather(S=S, P=P, y=y, title="Experiment with mistake")
    model_sales_mistake = lm("y ~ S*P", expt)
    summary(model_sales_mistake)
    contour_plot(model_sales_mistake)


def case_worksheet_5():
    """
    A = feed rate                           5 g/min     or   8 g/min
    B = initial inoculant amount            300 g       or 400 g
    C = feed substrate concentration        40 g/L      or  60 g/L
    D = dissolved oxygen set-point          4 mg/L      or   5 mg/L

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

    A, B, C, D = full_factorial(4, names=["A", "B", "C", "D"])

    A = supplement(A, name="Feed rate", units="g/min", lo=5, high=8.0)
    B = supplement(B, name="Initial inoculant amount", units="g", lo=300, hi=400)
    C = supplement(C, name="Feed substrate concentration", units="g/L", lo=40, hi=60)
    D = supplement(D, name="Dissolved oxygen set-point", units="mg/L", lo=4, hi=5)

    y = c(
        60,
        59,
        63,
        61,
        69,
        61,
        94,
        93,
        56,
        63,
        70,
        65,
        44,
        45,
        78,
        77,
        units="g/L",
        name="Yield",
    )

    expt = gather(A=A, B=B, C=C, D=D, y=y, title="Initial experiments; full factorial")
    model_start = lm("y ~ A*B*C*D", expt)

    summary(model_start)
    # pareto_plot(model_start, plot_width=800)
    contour_plot(model_start, "A", "B")
    contour_plot(model_start, "B", "C")
    contour_plot(model_start, "C", "D")


def api_usage():
    A, B, C, D = full_factorial(4, names=["A", "B", "C", "D"])

    A = supplement(A, name="Feed rate", units="g/min", lo=5, high=8.0)
    B = supplement(B, name="Initial inoculant amount", units="g", lo=300, hi=400)
    C = supplement(C, name="Feed substrate concentration", units="g/L", lo=40, hi=60)
    D = supplement(D, name="Dissolved oxygen set-point", units="mg/L", lo=4, hi=5)

    expt = gather(A, B, C, D, title="Initial experiments; full factorial")

    expt.show_actual(std_order=True)
    expt.show_actual(random_order=True, seed=13)
    expt.show_coded()
    expt.power()
    expt.export(save_as="xls", filename="abc.xlsx")

    center_points = expt.get_center_points()
    expt.append(center_points)
    cp_expt = 70
    expt["y"] = c(
        60,
        59,
        63,
        61,
        69,
        61,
        94,
        93,
        56,
        63,
        70,
        65,
        44,
        45,
        78,
        77,
        cp_expt,
        units="g/L",
        name="Yield",
    )

    # model_start = lm("y ~ <full>", expt)
    # model_start = lm("y ~ <2fi>", expt)
    # model_start = lm("y ~ <3fi>", expt)


def case_worksheet_6():
    """Half-fraction"""
    # The half-fraction, when C = A*B
    A = c(-1, +1, -1, +1)
    B = c(-1, -1, +1, +1)
    C = A * B

    # The response: stability [units=days]
    y = c(41, 27, 35, 20, name="Stability", units="days")

    # Linear model using only 4 experiments
    expt = gather(A=A, B=B, C=C, y=y, title="Half-fraction, using C = A*B")
    model_stability_poshalf = lm("y ~ A*B*C", expt)
    summary(model_stability_poshalf)
    pareto_plot(model_stability_poshalf)

    # The half-fraction, when C = -A*B
    C = -A * B
    y = c(41, 27, 35, 20, name="Stability", units="days")

    # Linear model using only 4 experiments
    expt = gather(A=A, B=B, C=C, y=y, title="Half-fraction, using C = -A*B")
    model_stability_neghalf = lm("y ~ A*B*C", expt)
    summary(model_stability_neghalf)
    pareto_plot(model_stability_neghalf)


def case_worksheet_8():
    """Highly-fractionated factorial"""
    A, B, C = full_factorial(3, names=["A", "B", "C"])

    # These 4 factors are generated, using the trade-off table relationships
    D = A * B
    E = A * C
    F = B * C
    G = A * B * C

    # These are the 8 experimental outcomes, corresponding to the 8 entries
    # in each of the vectors above
    y = c(320, 276, 306, 290, 272, 274, 290, 255)

    expt = gather(A=A, B=B, C=C, D=D, E=E, F=F, G=G, y=y, title="Fractopm")

    # And finally, the linear model
    mod_ff = lm("y ~ A*B*C*D*E*F*G", data=expt)
    summary(mod_ff)
    pareto_plot(mod_ff)

    # Now rebuild the linear model with only the 4 important terms
    mod_res4 = lm("y ~ A*C*E*G", data=expt)
    summary(mod_res4)
    pareto_plot(mod_res4)


def case_worksheet_9():
    """
    Experiment
    number       Date and time        Duration    Product created [g]
                                      [hours]      per unit sugar used
    1            06 May 2019 09:36    24.0        23
    2            06 May 2019 09:37    48.0        64
    3            06 May 2019 09:38    36.0        51
    4            06 May 2019 09:38    36.0        54
    5            06 May 2019 09:40    60.0        71
    6            20 May 2019 10:33    75.0        79
    7            20 May 2019 10:40    90.0        81
    8            20 May 2019 10:44    95.0        82
    9            20 May 2019 10:45    105.0       67
    """
    d1 = c(24, 48, center=36, range=(24, 48), coded=False, units="hours", name="Duration")
    D1 = d1.to_coded()
    y1 = c(23, 64, name="Production", units="g/unit sugar")
    expt1 = gather(D=D1, y=y1, title="Starting off")
    model1 = lm("y ~ D", data=expt1)
    summary(model1)
    p = plot_model(model1, "D", "y", xlim=(-2, 5), color="blue")

    d2 = d1.extend([36, 36])
    D2 = d2.to_coded()
    y2 = y1.extend([51, 54])
    expt2 = gather(D=D2, y=y2, title="Added 2 center points")

    # Model 2: y = intercept + D
    model2 = lm("y ~ D", data=expt2)
    summary(model2)
    p = plot_model(model2, "D", "y", fig=p, xlim=(-2, 5), color="darkgreen")

    # Model 2B: y = intercept + D + D^2
    model2B = lm("y ~ D + I(D**2)", data=expt2)
    summary(model2B)
    p = plot_model(model2B, "D", "y", fig=p, xlim=(-2, 5), color="red")

    # Try a new point at +2:
    d3 = d2.extend([60])
    D3 = d3.to_coded()
    predict(model2B, D=D3)  # predicts ___

    # Actual y = 71. Therefore, our model isn't so good. Improve it:
    y3 = y2.extend([71])
    expt3 = gather(D=D3, y=y3, title="Extend out to +2 (coded)")
    model3 = lm("y ~ D + I(D**2)", data=expt3, name="Quadratic model")
    summary(model3)

    # Plot it again: purple
    p = plot_model(model3, "D", "y", fig=p, xlim=(-2, 5), color="purple")

    # ------------
    # Normally at this point I would reset the frame of reference;
    # keep our range the same (24 hours)
    # -1: 36 hours [prior coded "0" now becomes "-1]
    #  0: 48 hours [prior coded "+1" now becomes "0"]
    # +1: 60 hours [prior coded "+2" now becomes "+1"]

    # At this point I will also switch the model type back to real-world units.
    # For 2 reasons: there is only 1 variable that is interesting (duration), so
    # coded units don't matter; secondly the model types we will use cannot be
    # built when the coded x-value is negative.
    d4 = c(24, 48, 36, 36, 60, coded=False, units="hours", name="Duration")
    y4 = c(23, 64, 51, 54, 71, name="Production", units="g/unit sugar")

    # Rebuild the model, and start the plots again. Try different model type
    expt4 = gather(d=d4, y=y4, title="Switch over to real-world units")
    model4 = lm("y ~ d + I(d**2)", data=expt4)
    summary(model4)
    p = plot_model(model4, "d", "y", xlim=(20, 105), color="purple")

    # Let's see how well this model fits. If we run an experiment at 75 hours,
    # we should notice a drop-off. But, predict it first:
    d5 = d4.extend([75])
    print(predict(model4, d=d5))

    # Expect a predicted value of 57 in the output. # Actual result: 79.
    # Bad prediction.

    # Let's try a different model structure: y = intercept + 1/d
    y5 = y4.extend([79])
    expt5 = gather(d=d5, y=y5, title="Hyperbolic model")
    model5 = lm("y ~ I(1/d)", data=expt5)
    summary(model5)
    p = plot_model(model5, "d", "y", fig=p, xlim=(20, 105), color="darkgreen")

    # -------
    # Let's try at around 90 hours. Expect an outcome of around 83 units.
    d6 = d5.extend([90])
    print(predict(model5, d=d6))

    # Got a value of 81 instead. Stabilized?
    y6 = y5.extend([81])
    expt6 = gather(d=d6, y=y6, title="Hyperbolic model with point at d=90 hrs")
    # Rebuild the model with the new data point, keeping model structure: y=1/d
    model6 = lm("y ~  I(1/d)", data=expt6)
    summary(model6)
    p = plot_model(model6, "d", "y", fig=p, xlim=(20, 105), color="blue")

    # Try adding few more points:
    # Point 8. Try 95 hours. Seems to sqrt.  Predict 83.5; actual: 82
    # Point 9: Try 105 hour to see decline.  Predict 85.4; actual: 67
    # Massive difference: something is different in the system here?

    ## Build a model with all values now
    d7 = d6.extend([95, 105])
    print(predict(model6, d=d7))

    y7 = y6.extend([82, 67])
    expt7 = gather(d=d7, y=y7, title="Hyperbolic model all 9 points")
    model7 = lm("y ~  I(1/d)", data=expt7)
    summary(model7)
    p = plot_model(model7, "d", "y", fig=p, xlim=(20, 110), color="red")

    # The current model structure does not allow for decrease/stabilization.
    # Rebuild it with a different structure
    # Model 7:  y = 1/d               SE = 6.181
    # Model 8:  y = d + 1/sqrt(d)     SE = 4.596
    # Model 9:  y = d + 1/log(d)      SE = 4.648
    # Model 10: y = d + d^2           SE = 4.321

    model8 = lm("y ~ d + I(1/np.sqrt(d))", data=expt7, name="With sqrt term")
    summary(model8)
    p = plot_model(model8, "d", "y", fig=p, xlim=(20, 110), color="orange")

    model9 = lm("y ~ d + I(1/np.log(d))", data=expt7, name="With log term")
    summary(model9)
    p = plot_model(model9, "d", "y", fig=p, xlim=(20, 110), color="brown")

    model10 = lm("y ~ d + I(d**2)", data=expt7, name="With quadratic term")
    summary(model10)
    p = plot_model(model10, "d", "y", fig=p, xlim=(20, 110), color="darkcyan")


def issue20():
    d4 = c(24, 48, 36, 36, 60, units="hours", lo=24, high=48)
    y4 = c(31, 65, 52, 54, 69)
    expt4 = gather(d=d4, y=y4, title="RW units")
    model4 = lm("y ~ d + I(np.power(d, 2))", data=expt4)
    summary(model4)


def case_worksheet_10():
    # Price: 0 # 0.25 above and 0.25 $/part below
    p = c(
        0.75,
        0.75,
        0.65,
        0.85,
        0.65,
        0.85,
        center=0.75,
        range=[0.65, 0.85],
        name="Price",
        units="$/part",
    )
    t = c(
        325,
        325,
        250,
        250,
        400,
        400,
        center=325,
        range=[250, 400],
        name="Throughput",
        units="parts/hour",
    )
    P1 = p.to_coded()
    T1 = t.to_coded()
    y1 = c(
        7740,
        7755,
        5651,
        5812,
        7363,
        7397,
        name="Response: profit per hour",
        units="$/hour",
    )
    expt1 = gather(P=P1, T=T1, y=y1, title="First experiment")

    mod_base1 = lm("y ~ P * T", data=expt1)
    summary(mod_base1)
    contour_plot(mod_base1, "P", "T", show=False)

    # Predict the points, using the model:
    prediction_1 = predict(mod_base1, P=P1, T=T1)
    print(prediction_1)
    print(y1 - prediction_1)

    # We see clear non-linearity, especially when viewed in the direction of T

    # Try anyway to make a prediction, to verify it
    # P ~ 0.15 and T ~ 2.0:
    P2 = P1.extend([0.15])
    T2 = T1.extend([2.0])
    p2 = P2.to_realworld()
    t2 = T2.to_realworld()
    print(t2)  # 0.765
    print(p2)  # 475
    print(predict(mod_base1, P=P2, T=T2))

    # Should have a predicted profit of 8599, but actual is 4654.
    # Confirms our model is in a very nonlinear region in the T=Throughput
    # direction.

    # Perhaps our factorial was far too big. Make the range smaller in T.
    # Prior range = [250;400]; now try [287.5; ]

    # Second factorial: re-use some of the points
    # * Original center point become bottom left
    # * Original (+1, +1) become top right
    p3 = c(
        0.75,
        0.85,
        0.75,
        0.85,
        0.65,
        0.85,
        0.765,
        center=0.80,
        range=[0.75, 0.85],
        name="Price",
        units="$/part",
    )
    t3 = c(
        325,
        325,
        400,
        400,
        400,
        250,
        475,
        center=(325 + 400) / 2,
        range=(325, 400),
        name="Throughput",
        units="parts/hour",
    )

    # 2nd,
    y3 = c(
        7755,
        7784,
        7373,
        7397,
        7363,
        5812,
        4654,
        name="Response: profit per hour",
        units="$/hour",
    )
    P3 = p3.to_coded()
    T3 = t3.to_coded()
    expt3 = gather(P=P3, T=T3, y=y3, title="Smaller ranges")
    mod_base3 = lm("y ~ P * T", data=expt3)
    summary(mod_base3)
    contour_plot(mod_base3, "P", "T")

    # Predict directly from least squares model, the next experiment
    # at coded values of (+2, +2) seems good
    predict(mod_base3, P=+2, T=+2)
    # Prediction is 7855

    # In RW units that corresponds to: p=0.9 and t=437.5 = 438 parts/hour
    P4 = P3.extend([+2])
    T4 = T3.extend([+2])
    print(P4.to_realworld())
    print(T4.to_realworld())

    # ACTUAL value achieved is 6325. Not a good prediction yet either.
    # Add this point to the model. This point is below any of the base factorial
    # points!
    y4 = y3.extend([6325])
    expt4 = gather(P=P4, T=T4, y=y4, title="Adding the next exploration")
    mod_base4 = lm("y ~ P * T", data=expt4)
    contour_plot(mod_base4, "P", "T")

    # It is clear that this model does not meet our needs. We need a model with
    # quadratic fitting, nonlinear terms, to estimate the nonlinear surface.
    expt5 = expt4.copy()
    mod_base5 = lm("y ~ P*T + I(P**2) + I(T**2)", data=expt5)
    print(summary(mod_base5))

    # add the xlim input in a second round
    contour_plot(mod_base5, "P", "T", xlim=(-2, 4))

    # Run at (P=3, T=-0.3) for the next run
    P6 = P4.extend([+3])
    T6 = T4.extend([-0.3])
    print(P6.to_realworld())
    print(T6.to_realworld())

    # Corresponds to p = 0.95 $/part, t=351 parts/hour
    # Predict = 7939
    # Actual = 7969. Really good matching.
    # UPdate the model and check
    y6 = y4.extend([7969])
    expt6 = gather(P=P6, T=T6, y=y6, title="After extrapolation, based on quadratic term")
    mod_base6 = lm("y ~ P*T + I(P**2) + I(T**2)", data=expt6)
    contour_plot(mod_base6, "P", "T", xlim=(-2, 5))

    # Extrapolate again to (P=5, T=-0.3) for the next run
    P7 = P6.extend([+5])
    T7 = T6.extend([-0.3])
    print(P7.to_realworld())
    print(T7.to_realworld())
    predict(mod_base6, P=5, T=-0.3)

    # to P = 1.05, T=351 parts/hour
    # Predict = 7982
    # Actual = 8018. Better than predicted. Perhaps surface is a steeper quadratic.
    # Update the model and check
    y7 = y6.extend([7982])
    expt7 = gather(P=P7, T=T7, y=y7, title="With 2 extrapolations")
    mod_base7 = lm("y ~ P*T + I(P**2) + I(T**2)", data=expt7)
    contour_plot(mod_base7, "P", "T", xlim=(-2, 148))


def case_worksheet_10C():
    # Price: 0 # 0.05 above and 0.05 $/part below
    p1 = c(
        0.75,
        0.75,
        0.7,
        0.8,
        0.7,
        0.80,
        center=0.75,
        range=[0.70, 0.80],
        name="Price",
        units="$/part",
    )
    t1 = c(
        325,
        325,
        300,
        300,
        350,
        350,
        center=325,
        range=[300, 350],
        name="Throughput",
        units="parts/hour",
    )
    P1 = p1.to_coded()
    T1 = t1.to_coded()
    y1 = c(
        7082,
        7089,
        6637,
        6686,
        7181,
        7234,
        name="Response: profit per hour",
        units="$/hour",
    )
    expt1 = gather(P=P1, T=T1, y=y1, title="First experiment")

    mod_base1 = lm("y ~ P * T", data=expt1)
    summary(mod_base1)
    contour_plot(mod_base1, "P", "T")

    # Predict the points, using the model:
    prediction_1 = predict(mod_base1, P=P1, T=T1)
    print(prediction_1)
    print(y1 - prediction_1)

    # We see clear non-linearity, especially when viewed in the direction of T

    # Try anyway to make a prediction, to verify it
    # P ~ 0.7 and T ~ 2.0:
    P2 = P1.extend([0.7])
    T2 = T1.extend([2.0])
    p2 = P2.to_realworld()
    t2 = T2.to_realworld()
    print(p2)  # 0.785
    print(t2)  # 375
    print(predict(mod_base1, P=P2, T=T2))

    # Should have a predicted profit of 7550, but actual is 7094.
    # Confirms our model is in a very nonlinear region in the T=Throughput
    # direction.

    # Add axial points, starting in the T direction:
    P3 = P2.extend([0, 0])
    T3 = T2.extend([1.68, -1.68])
    p3 = P3.to_realworld()
    t3 = T3.to_realworld()
    print(p3)  # 0.75, 0.75
    print(t3)  # 367, 283

    # Now build model with quadratic term in the T direction
    y3 = y1.extend([7094, 7174, 6258])
    expt3 = gather(P=P3, T=T3, y=y3, title="With axial points")
    mod_base3 = lm("y ~ P * T + I(T**2)", data=expt3)
    summary(mod_base3)
    contour_plot(mod_base3, "P", "T", xlim=(-1.5, 5))
    #

    # Try extrapolating far out: (P, T) = (4, 1)
    P4 = P3.extend([4])
    T4 = T3.extend([1])
    p4 = P4.to_realworld()
    t4 = T4.to_realworld()
    print(p4)  # 0.95
    print(t4)  # 350

    predict(mod_base3, P=P4, T=T4)  # 7301
    # Actual: 7291  # great! Keep going
    y4 = y3.extend([7291])

    # Try extrapolating far out: (P, T) = (6, 1)
    P5 = P4.extend([6])
    T5 = T4.extend([1])
    p5 = P5.to_realworld()
    t5 = T5.to_realworld()
    print(p5)  # 1.05
    print(t5)  # 350

    predict(mod_base3, P=P5, T=T5)  # 7344
    # Actual: 7324  # great! Keep going
    y5 = y4.extend([7324])

    # Visualize model first
    expt5 = gather(P=P5, T=T5, y=y5, title="With extrapolated points")
    mod_base5 = lm("y ~ P * T + I(T**2)", data=expt5)
    summary(mod_base5)
    contour_plot(mod_base5, "P", "T", xlim=(-1.5, 18))

    # Try extrapolating further out: (P, T) = (10, 1)
    P6 = P5.extend([10])
    T6 = T5.extend([1])
    p6 = P6.to_realworld()
    t6 = T6.to_realworld()
    print(p6)  # 1.25
    print(t6)  # 350

    predict(mod_base3, P=P6, T=T6)  # 7431
    # Actual: 7378  # Not matching; rebuild the model eventually.
    _ = y5.extend([7378])


def case_worksheet_10B():
    # Code for this system: https://rsmopt.com/system/concrete-strength/

    # C: cement = 1.8 and 4.2 kg)
    # W: amount of water (between 0.4 and 1.1 L)

    c1 = c(2.5, 3, 2.5, 3, center=2.75, range=[2.5, 3], name="cement", units="kg")
    w1 = c(
        0.5,
        0.5,
        0.9,
        0.9,
        center=0.7,
        range=[0.5, 0.9],
        name="Throughput",
        units="parts/hour",
    )
    C1 = c1.to_coded()
    W1 = w1.to_coded()
    y1 = c(14476, 14598, 14616, 14465, name="Strength", units="-")
    expt1 = gather(C=C1, W=W1, y=y1, title="First experiment")

    mod_base1 = lm("y ~ C * W", data=expt1)
    summary(mod_base1)
    contour_plot(mod_base1, "C", "W")

    # Predict the points, using the model:
    prediction_1 = predict(mod_base1, C=C1, W=W1)
    print(prediction_1)
    print(y1 - prediction_1)

    # Very nonlinear: saddle: up left, or bottom right
    # Bottom right: (C, W) = (2, -2)
    C2 = C1.extend([2])
    W2 = W1.extend([-2])

    # Predict at this point: 14794
    predict(mod_base1, C=C2, W=W2)
    c2 = C2.to_realworld()
    w2 = W2.to_realworld()

    # Actual: at c=3.25; w=0.4 (constraint): 14362. So wrong direction
    y1 = c(14476, 14598, 14616, 14465, name="Strength", units="-")
    expt1 = gather(C=C1, W=W1, y=y1, title="First experiment")

    # Try the other way: C, W= -2, 2
    C2 = C1.extend([-2])
    W2 = W1.extend([+2])

    # Predict at this point: 14830
    predict(mod_base1, C=C2, W=W2)
    c2 = C2.to_realworld()  # 2.25
    w2 = W2.to_realworld()  # 1.1
    print(c2, w2)
    # Actual: 13982;


if __name__ == "__main__":
    # tradeoff_table()
    # case_3B()    # case_3C(show=True)
    # case_3D()
    # case_worksheet_5()
    # api_usage()
    # case_worksheet_6()
    # case_worksheet_8()
    # case_worksheet_9()
    # case_worksheet_10()
    case_worksheet_10C()

    # case_w2()
    # case_w4_1()
    # case_w4_2()
