- **`PLS.select_n_components` runs about twice as fast on wide data.** The per-row and per-column sums of squares
  that PLS and PCA fits keep at every component were computed one row or column at a time in Python; they are now one
  vectorised call. Fitted values are unchanged (under 1e-12 relative), and cross-validated PRESS to 1e-9: 7-fold
  cross-validation of the 650-column tablet spectra, 10 repeats to 20 components, takes 17 s, where it took 31 s.
