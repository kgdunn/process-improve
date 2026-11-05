- **`sensory.mixed_assessor_model` says which cells are missing, and why an F-test is NaN.** For an
  attribute where a panelist did not score every product, the warning named only the attribute.
  It now names each such panelist and the products they did not score, and the same text reaches
  `sensory_panel_check` and `sensory_analyze_descriptive` under `library_warnings`. The `ftests`
  table has a new `note` column that says why a row's F-tests are NaN: the missing cells, a panel
  too small to test (one product, one panelist, or two products for the MAM F-test), or an error
  sum of squares of zero. It is empty when both F-tests are computed. (#1099)
