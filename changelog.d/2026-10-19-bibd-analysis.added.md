- **`bibd_analysis`: the intrablock, interblock and combined treatment estimates of a balanced
  incomplete block design** (`process_improve.experiments`, #789 B7). From the runs it finds the
  design's a, b, k, r and lambda (and refuses a layout that is not balanced), and gives the adjusted
  treatment totals Q, the intrablock estimates kQ/(lambda a), the adjusted block totals Q', Yates's
  interblock estimates from the block totals, and the combined estimates that weight the two by
  the error and block variances, with the adjusted and unadjusted sums of squares. It reproduces
  Montgomery's Example 4.5 (Q = -9/3 to 20/3, interblock estimates 10.50, -3.50, -0.50 and -6.50,
  combined -1.09, -0.88, -0.50 and 2.47), Box, Hunter and Hunter's T, B and Q of Table 8D.4, and
  Davies's tyre and rubber totals.
