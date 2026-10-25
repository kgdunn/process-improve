- **The reference catalogue holds Box and Draper (2007), *Response Surfaces, Mixtures, and Ridge Analyses*
  (2nd edition), the whole book.** `box-draper-2007` holds 177 experiments and 145 designs from chapters 1 to 20
  and the answers to the exercises, with 12,499 printed numbers: least squares and its matrix algebra, two-level
  factorials and fractions, steepest ascent, second-order fits and their canonical analysis, transformations of the
  response, ridge analysis, composite, Box-Behnken and other response surface designs, variance and bias of designs,
  and mixture experiments. 8,724 checks agree with the library. 1,040 do not, and each says why: an error in the
  book, the book's own rounding, a convention the library does not follow, or a library bug. 3,223 numbers are not
  recomputed, and each names what the library lacks, such as ridge analysis under linear restrictions, iteratively
  reweighted least squares, the bias of a design, and mixture lattices of degree 4 and more.
- **New check options in the reference catalogue:** `exp10` reads a fit to common logs back as an antilog;
  `sign_of` and `abs` on `library_value` give an eigenvector the sign a source prints; `ss_total_uncorrected`,
  `ss_mean`, `ss_regression_uncorrected`, `df_model_uncorrected` and `ms_regression_uncorrected` check an analysis
  of variance about zero; the `angular_degrees` summary is arcsin (m/n)^1/2 in degrees, with a source's own values
  at m = 0 and m = n.
