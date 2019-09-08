# (c) Kevin Dunn, 2018.

from numpy.random import normal

def grocery(p=3.46, h=150, P=None, H=None):
    """
    Simulates a grocery store profit function where there are 2 factors:
    * p = selling price of the product, measured in dollars and cents
    * h = height of the product on the shelf, measured in centimeters above
          the ground.

    Typical values are p = $3.50 and h = 150cm
    The outcome is: profit made per hour [dollars/hour].
    """
    if P is None:
        P = p
    if H is None:
        H = h

    if ( (len(P) > 1) | (len(H) > 1)):
        assert False, ("Running the grocery store experiments in parallel is "
                       "(intentionally) not allowed.")

    #if (not all(is.finite(P)) | !all(is.finite(H))) {
    #stop("All function inputs must be finite numbers.")
    #} else if (P < 0) {
    #stop("Please provide a positive sales price, P.")
    #} else if (H < 0) {
    #stop("The height of the shelving, H, must be a positive value.")
    #} else{
    a_coded = (P - 3.2) / 0.2
    b_coded = (H - 50) / 100
    y = round( (18 * a_coded + 12 * b_coded - 7 * a_coded * a_coded
                 - 6.0 * b_coded * b_coded  - 8.5 * a_coded * b_coded + 60) * 10.0
                + normal(0, 1) * 2
               )
    return y
