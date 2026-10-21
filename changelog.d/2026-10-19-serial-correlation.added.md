- **`fit_ma1` and `difference_variance`: a first-order moving average, and the variance of a
  difference of averages of serially correlated observations** (`process_improve.univariate`,
  #802 G3). `fit_ma1` fits y_t = mu + e_t - theta e_{t-1} to a record in time order by matching
  its lag-1 autocorrelation and variance, as Box, Hunter and Hunter do (theta 0.32 and a residual
  variance of 7.5 for their 210 batch yields), or by maximum likelihood (`method="likelihood"`),
  which uses the whole record and gives a standard error. `difference_variance` gives the variance
  of the difference of two averages of n observations with a lag-1 autocorrelation, for adjacent
  averages (2 sigma^2/n)(1 + rho1 (2n - 3)/n), which is the book's 1 + 1.7 rho1 for averages of 10:
  0.84 for the 210 yields, against 1.66 if they were independent.
