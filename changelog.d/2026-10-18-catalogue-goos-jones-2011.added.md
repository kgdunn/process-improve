- **The reference catalogue reads Goos and Jones, *Optimal Design of Experiments: A Case Study Approach*
  (2011), chapters 1 and 2.** It holds:
  - the splits of a two-machine comparison's runs (Tables 1.1 to 1.4), checked with `allocate_runs`;
  - the GeneBe extraction experiment, with its fits, predictions, alias matrix (Table 2.7) and variance
    inflation factors;
  - the four-run design of Table 2.8 and the D-optimal design of Table 2.11;
  - the worked coordinate-exchange example, each design compared on |X'X|.

  394 printed numbers give 424 checks, all agreeing or explained. The 11 explained disagreements are
  misprints:
  - two t ratios in Table 2.5;
  - the time coefficient of Equation 2.1;
  - Table 2.12's VIF of 1.25 for four terms, which Table 2.11's runs, and the best design a coordinate
    exchange finds, give as 1.125.

  New check options: an array compared as arithmetic (`expression`, for an alias matrix printed in
  thirds), a product column named by its columns (`of`), the mean of a product column (`mean`), the
  square root of a library value (`root`), and the first columns of a library design (`design_columns`).
