- **Crossed inner and outer arrays for robust design.** `cross_arrays(inner, outer)` runs every
  run of an inner array (the controllable factors) at every run of an outer array (the noise
  factors): one row per pair, ordered by the inner run and then the outer run, with `inner_run`
  and `outer_run` labels to lay the readings out as the sources print them. It takes a
  `DesignResult`, in coded or actual units, or a DataFrame. It builds Ullmann's Table 8 (11 x 4 =
  44 runs), and with Montgomery's readings it gives the leaf springs' means and variances of
  Table 12.1.
