- **`evop_worksheet`: the evolutionary-operation calculation sheet of a 2^2 design with a centre
  point** (`process_improve.monitoring`, #802 G1). For each cycle it gives each point's sums and
  averages, the effects of the two factors, their interaction and the change in mean, the
  differences between each point's previous average and its new reading, their range, the
  standard deviation from it, S = range x f_{k,n}, with its running sum and average, and the
  two-standard-error limits of the averages, the effects and the change in mean; the last sheet is
  the information board. `evop_f(k, n)` gives f_{k,n} = sqrt((n - 1)/n)/d2(k), and `range_d2(k)`
  the mean range of k standard normal observations. The sheet reproduces Montgomery's Example 11.7
  (S 0.60 and 0.56, limits 0.85 and 0.67 for the effects, 0.76 and 0.60 for the change in mean).
