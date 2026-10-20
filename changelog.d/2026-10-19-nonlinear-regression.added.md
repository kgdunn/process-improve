- **`nonlinear_regression`: least squares for a model nonlinear in its parameters, with the usual
  inference** (`process_improve.regression`, #789 B8). `nonlinear_regression(model, x, y, start)`
  fits y = model(x, *theta) by least squares from the named starting values, and gives each
  parameter's estimate, standard error, t value, p value and interval from the linearization at the
  estimate, the correlations of the estimates, the residual sum of squares and mean square, the
  fitted values and residuals, and the contour S(theta) = S_R (1 + p/(n - p) F) that bounds the
  joint confidence region. It reproduces Box, Hunter and Hunter's biochemical oxygen demand fit
  (section 14.6): b1 = 213.8, S_R = 1168.01, and the 90% contour 3690.91 with the table's F of 4.32
  (3693.57 with the exact F). Parameters whose derivatives are linearly dependent are refused.
