- **Bartlett's test, the Kruskal-Wallis test, a variance's chi-square interval, and the power and
  sample size of the one-way analysis of variance (#789 B5).** In `process_improve.univariate`:
  - `bartlett_test(*groups)`, with its steps `q` and `c` as Montgomery prints them (Example 3.4:
    0.43, P = 0.933);
  - `kruskal_wallis_test(*groups)`, the analysis of variance of the ranks, allowing for ties
    (H = 16.91, P = 7.38e-4);
  - `variance_interval(variance, df)`, also on fractional degrees of freedom, with the interval of
    the standard deviation;
  - `one_way_power(n, sigma, means)` from the noncentral F, with the parameter of Montgomery's
    operating characteristic curves, and `one_way_sample_size(power, sigma, means)`, the smallest
    number of runs in each group that reaches a power; `least_favourable_means(difference,
    n_groups)` gives the means at which a difference is hardest to detect, as Minitab works it
    (0.804838 and 0.915384 for five and six wafers; 8 samples of each mortar for 0.95).
