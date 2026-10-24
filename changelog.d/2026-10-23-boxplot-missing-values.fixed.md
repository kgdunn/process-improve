- **The `boxplot` tool leaves missing values out of each box (#513).** One missing value made every quartile
  of its box NaN, so the box came back as five `null`s and drew as nothing, with no error. The quartiles,
  whiskers and outliers are now computed from the values present. A column or group with no values at all
  is refused by name, as `{"error": "'y' has no values to plot: every value is missing."}`.
