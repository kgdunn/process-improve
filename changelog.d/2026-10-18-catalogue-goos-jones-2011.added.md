- **The reference catalogue reads Goos and Jones, *Optimal Design of Experiments: A Case Study Approach*
  (2011), chapters 1 to 4.** It holds:
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
    equations of the final model, one per supplier.

  774 printed numbers give 845 checks, all agreeing or explained. The 24 explained disagreements are
  misprints or numbers read off a figure:
  - two t ratios in Table 2.5 and two in Table 3.7;
  - the time coefficient of Equation 2.1;
  - Table 2.12's VIF of 1.25 for four terms, which Table 2.11's runs, and the best design a coordinate
    exchange finds, give as 1.125;
  - the block row of Equation 3.14 and the determinant printed under it;
  - an RMSE "more than one and a half" for the reduced model of the 20 runs, whose RMSE is 1.07;
  - chapter 4's two determinants, whose ratio gives a relative D-efficiency of 0.952, not 93%;
  - supplier 3's speed coefficient, -0.0079 to the four decimals printed as -0.0080;
  - three quantiles of chapter 4's FDS plot, read off Figure 4.1.

  New check kinds: the corner of the cube where a model predicts the most (`best_corner`), a model's
  AICc (`aicc`), and one design's D- or I-efficiency relative to another entry's
  (`relative_efficiency`). New check options: an array compared as arithmetic (`expression`, for an
  alias matrix printed in thirds), a product column named by its columns (`of`), the mean of a
  product column (`mean`), the square root of a library value (`root`), the first columns of a
  library design (`design_columns`), and some of the printed runs compared with a library design
  (`runs`, for one supplier's runs).
