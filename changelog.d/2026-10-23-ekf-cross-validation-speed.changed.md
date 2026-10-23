- **`PCA.select_n_components` with the element-wise k-fold scheme (`cv_scheme="ekf"`, the default) runs about four
  times faster on wide data.** Each EM step now projects onto the leading components from the smaller Gram matrix
  instead of computing a full SVD. The PRESS values are unchanged to rounding (1e-13 relative): 7-fold cross-validation
  of the tablet spectra up to 20 components takes 103 s, where it took 445 s.
