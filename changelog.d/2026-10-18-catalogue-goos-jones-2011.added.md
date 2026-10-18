- **The reference catalogue reads Goos and Jones, *Optimal Design of Experiments: A Case Study Approach*
  (2011), the whole book.** It holds:
  - the splits of a two-machine comparison's runs (Tables 1.1 to 1.4), checked with `allocate_runs`;
  - the GeneBe extraction experiment, with its fits, predictions, alias matrix (Table 2.7) and variance
    inflation factors;
  - the four-run design of Table 2.8 and the D-optimal design of Table 2.11;
  - the worked coordinate-exchange example, each design compared on |X'X|;
  - the extraction experiment augmented with eight runs (chapter 3): the nine competing models of
    Table 3.1 with their AICc, the settings each recommends (Table 3.2), the power and variances of the
    14-term model, and the fits of the 20 runs;
  - the peel strength experiment (chapter 4): its D-optimal and I-optimal designs, compared on their
    relative variances, average prediction variance, FDS plots and each other, and the prediction
    equations of the final model, one per supplier;
  - the yield of a reaction (chapter 5): a 3 x 3 factorial with its quadratic fit, and a 15-run
    D-optimal design in a constrained region with its cubic fits, lack-of-fit test and greatest
    predicted yield;
  - the rolling mill (chapter 6), a mixture with two process variables: its design's balance and
    prediction variance, its first-order Scheffé fit in pseudocomponents, and the half fraction of the
    2^5 factorial that the mixture constraint rules out;
  - the pastry dough experiment (chapter 7), a D-optimal response surface design in seven days of four
    runs: its efficiency factors and the average row of each day's model matrix, and REML fits with
    the days as random effects, beside the orthogonally blocked central composite design it replaces;
  - the stability of riboflavin (chapter 8), a D-optimal screening design in eight days of four: its
    blocking VIFs beside a random design's and Dr. Xu's blocked half fraction, the 64-run
    alternatives, and the fixed and random block effects analyses;
  - the polypropylene experiment (chapter 9), 18 samples chosen from 40 in stock with their EPDM,
    ethylene and EVA as covariates: the design's balance, its relative standard errors beside a design
    that treats the covariates as ordinary factors, the main-effects fit with its gas type coded as
    JMP codes it, the prediction equations and the points of Figure 9.4's contours, and two run orders
    of the 2^3 factorial robust to a linear time trend, with their trend-robustness (Equation 9.19);
  - the wind tunnel experiment (chapter 10), a 50-run I-optimal split-plot design in ten whole plots:
    its variances, average and median prediction variance beside Dr. Cavendish's two designs, the
    ordinary and Kenward-Roger generalized least squares fits of the efficiency (Tables 10.5 to 10.7)
    and the simplified models of all five responses (Table 10.8), and the eight-whole-plot design of
    Tables 10.9 and 10.10;
  - the battery cell experiment (chapter 11), a 48-run D-optimal two-way split-plot design in 16 lots
    and six curing cycles: its variances and correlations of estimates with the rows and the columns
    as random effects (Tables 11.3 and 11.4), the Kenward-Roger fit (Table 11.6), the BLUPs of the
    rows and columns (Table 11.7) and the variance components, beside Pino's four-cycle plan and the
    12-run design of Tables 11.8 and 11.9.

  2164 printed numbers give 2361 checks, all agreeing or explained. The 77 explained disagreements are
  misprints, numbers read off a figure, or numbers of a design other than the one printed:
  - two t ratios in Table 2.5 and two in Table 3.7;
  - the time coefficient of Equation 2.1;
  - Table 2.12's VIF of 1.25 for four terms, which Table 2.11's runs, and the best design a coordinate
    exchange finds, give as 1.125;
  - the block row of Equation 3.14 and the determinant printed under it;
  - an RMSE "more than one and a half" for the reduced model of the 20 runs, whose RMSE is 1.07;
  - chapter 4's two determinants, whose ratio gives a relative D-efficiency of 0.952, not 93%;
  - supplier 3's speed coefficient, -0.0079 to the four decimals printed as -0.0080;
  - three quantiles of chapter 4's FDS plot, read off Figure 4.1;
  - Figure 7.1's two predictions at the settings it shows, which Table 7.9's models do not give;
  - the "about 3% better" of the 64-run design printed as D-optimal (Table 8.7), which holds the same
    runs as Table 8.6, blocked the same way;
  - ten relative standard errors of Table 9.4 that differ from the printed runs' in the fourth decimal
    (three of them inconsistent with Table 9.6's own standard errors);
  - Table 10.3's I-optimal column, the 89.2% relative D-efficiency and the 12.1% gain, which belong to
    an I-optimal design other than Table 10.2's (that design gives 91.4%); and the median prediction
    variance "about 0.47", 0.475 for Table 10.2;
  - the 45 degrees of freedom of Table 10.7's ordinary least squares, 35 for its 15-term model;
  - one p value of Table 10.8 (0.0013 for 0.00136) and one square in Table 10.11.

  Chapter 9's "95% prediction intervals" are the mean's 95% confidence intervals, and are checked as
  those; a new run's prediction interval is about twice as wide.

  New check kinds: the corner of the cube where a model predicts the most (`best_corner`), a model's
  AICc (`aicc`), and one design's D- or I-efficiency relative to another entry's
  (`relative_efficiency`). New check options: an array compared as arithmetic (`expression`, for an
  alias matrix printed in thirds), a product column named by its columns (`of`), the mean of a
  product column (`mean`), the square root of a library value (`root`), the first columns of a
  library design (`design_columns`), and some of the printed runs compared with a library design
  (`runs`, for one supplier's runs), the runs whose column is above or at least a value
  (`above` and `at_least` on `run_count`), a metric over the same metric of a second model
  (`relative_to`, for a VIF or efficiency factor due to blocking), in percent (`percent`) or counted
  (`count`), a block's average model row (`average`), and the split-plot analysis's coefficients,
  variance ratio and predictions, by Kenward and Roger's method (`df_method`, `information`), a
  design's runs evaluated as a split-plot design through their groups (`groups`, on
  `design_metric` and `relative_efficiency`), a relative efficiency in percent or as a gain (`percent`,
  `gain`), a run order's trend-robustness (`trend_robustness`, Equation 9.19), and a two-way
  split-plot analysis's crossed random effects, BLUPs and best corner (`crossed`, `blup`,
  `best_corner` on `split_plot`).
