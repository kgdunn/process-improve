- **`TPLS.required_inputs_` is removed.** It was the constant `{"F", "Y", "Z"}`, set in `__init__`,
  which sklearn's estimator convention does not allow, and read by nothing. (#1028)
