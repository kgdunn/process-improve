- **`box_tidwell`: Box and Tidwell's estimate of the power of a regressor**
  (`process_improve.regression`, #801 F2). For a response that is curved in a positive regressor
  but straight in a power of it, `box_tidwell(X, y, columns=None)` estimates the power of one or
  more regressors in y = b0 + sum_j b_j x_j^alpha_j (ln x at alpha = 0), with the other columns
  entering as they are. Each step regresses y on the regressors at the trial powers and on the
  constructed variables x^alpha ln x, and moves each power by gamma / beta; repeated, with a step
  halved when it would raise the residual sum of squares, this is the Gauss-Newton method for the
  least-squares estimate. The result gives each power, its standard error, the t value and p value
  of the constructed variable at power 1 (a test of a straight-line relation), the residual sums of
  squares at the estimate and at power 1, and the powers after each step. On Montgomery, Peck and
  Vining's windmill data the first step is the book's -0.92 and the estimate is -0.833 with a
  standard error of 0.159, as nonlinear least squares gives.
