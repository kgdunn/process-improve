- **`"model_selection"` on a mixture formula searches the formula's own terms.** A formula over blends
  alone that holds the constant, such as `"y ~ x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3 + x1:x2:x3 - 1"`, is
  analysed as a Scheffé model, but `"model_selection"` searched it as `"interactions"`: with an
  intercept the blends already make up, and without any term beyond the two-factor products. On
  Kurotori's propellant (Box and Draper, Table 17.1) it selected `y ~ 1`, with an `r_squared` of 0,
  where `"scheffe_special_cubic"` keeps the special cubic term. Every linear blending term now stays
  in, the search chooses among the formula's other terms, `candidate_model` is the formula, and there
  is no `note`. (#1129)
- **The conference-order warning of the minimal OMARS design names OMARS, at the caller's line.**
  `generate_design(factors, "omars")` for 21, 22, 33 or 34 factors warned, and noted in
  `metadata["note"]`, that the "definitive screening design" steps up its conference order, and the
  warning named a line in the library. It now says "OMARS design", and from `generate_design`,
  `dispatch_omars` or `dispatch_dsd` it points at the line that asked for the design. (#1129)
- **A mixture model's `"split_plot"` coefficients are labelled `coding: "actual"`.** They were labelled
  `"coded"`, although the components are proportions that have no coded scale, as `"coefficients"`
  already reports. (#1129)
- **The K-optimal search no longer lets the BLAS kernel choose between tied swaps.** At each step the candidate
  exchange behind `generate_design(..., "k_optimal")` took the first of the swaps that tie for the best
  condition number in the order the eigenvectors of `X'X` ranked them. An extreme eigenvalue is usually
  repeated on a grid, so those eigenvectors are any basis of its eigenspace, which the BLAS kernel's
  rounding picks, and the same `random_state` gave designs of different quality on AVX2 and AVX-512
  processors: a 3-factor interactions design in 10 runs had a condition number of 1.25 on one and 1.27 on
  the other. The first tied swap, by design run and then candidate, is now made. Over 9 cases and 5
  seeds, four OpenBLAS kernels then agree on every condition number, where one case-seed pair differed;
  on an AVX2 processor 8 of the 45 designs change, 6 to another design with the same condition number and
  2 from 1.25 to 1.27. (#1129)
