- **`"model_selection"` on a mixture formula searches the formula's own terms.** A formula over blends
  alone that holds the constant, such as `"y ~ x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3 + x1:x2:x3 - 1"`, is
  analysed as a Scheffé model, but `"model_selection"` searched it as `"interactions"`: with an
  intercept the blends already make up, and without any term beyond the two-factor products. On
  Kurotori's propellant (Box and Draper, Table 17.1) it selected `y ~ 1`, with an `r_squared` of 0,
  where `"scheffe_special_cubic"` keeps the special cubic term. Every linear blending term now stays
  in, the search chooses among the formula's other terms, `candidate_model` is the formula, and there
  is no `note`. (#1129)
- **The conference-order warning of the minimal OMARS design names OMARS, at the caller's line.**
  `generate_design(factors, "omars")` for 21, 22, 33 or 34 factors warned, and noted in
  `metadata["note"]`, that the "definitive screening design" steps up its conference order, and the
  warning named a line in the library. It now says "OMARS design", and from `generate_design`,
  `dispatch_omars` or `dispatch_dsd` it points at the line that asked for the design. (#1129)
- **A mixture model's `"split_plot"` coefficients are labelled `coding: "actual"`.** They were labelled
  `"coded"`, although the components are proportions that have no coded scale, as `"coefficients"`
  already reports. (#1129)
