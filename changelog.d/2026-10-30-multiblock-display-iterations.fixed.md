- **`MBPCA.display_results` and `MBPLS.display_results` print the iteration counts as integers.**
  Under numpy 2 the line read `Iterations per PC: [np.int64(10), np.int64(16)]`; it reads
  `[10, 16]` again, as does the debug log line of each fit. (#1028)
