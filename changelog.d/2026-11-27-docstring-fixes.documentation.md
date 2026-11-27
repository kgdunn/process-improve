- **Fix LaTeX math rendering in a few docstrings.** The raw docstrings of
  `multivariate.vip`, `multivariate.squared_cosine`,
  `multivariate.observation_contributions` and
  `bivariate._elbow_peak.calculate_line_length` had doubled backslashes
  (``\\sqrt``, ``\\frac``, ``\\sum``, ``\\text``) in their ``.. math::`` blocks and
  inline ``:math:`` roles. In a raw string that is two literal backslashes, which
  rendered as a LaTeX line break rather than the command, so the formulas came
  out broken. The commands now use a single backslash, matching the convention
  used in every other math block in the package.
