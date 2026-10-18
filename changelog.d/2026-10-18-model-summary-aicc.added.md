- **`analyze_experiment`'s model summary gives the corrected AIC.** `model_summary["aicc"]` is
  AIC + 2k(k + 1)/(n - k - 1), with k the model's coefficients and the error variance, as JMP prints
  it and the library's stepwise model selection scores a model. It is `None` when the runs are too
  few for the correction (n <= k + 1).
