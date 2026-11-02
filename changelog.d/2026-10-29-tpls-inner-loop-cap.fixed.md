- **The inner PLS loop of a `TPLS` fit stops at its cap of 500 iterations, not 502.** It counted from 0
  and stopped once the count exceeded the cap, so a component that never converged ran two extra
  iterations, and its debug log said it had converged. It now runs at most 500, and the log says how
  many it ran ("stopped after"). Components that converge are unchanged. (#1023)
