- **The batch mid-course correction no longer triggers osqp's deprecation warnings.** Its QP solve
  passed osqp's old `polish` setting name and left `raise_error` to its default, so every
  constrained correction gave a `DeprecationWarning` and a `PendingDeprecationWarning`, and the
  `evaluate_batch_control_policy` and `correct_batch_midcourse` tools reported both under
  `library_warnings`. The solve now uses `polishing` and passes `raise_error=False`, which every
  osqp release since 1.0 accepts; the results are unchanged.
