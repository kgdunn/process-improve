- **Messages, docstrings and documentation use American spelling** (#1232): "color", "center", "analyze",
  "optimize", "behavior", "catalog", "labeled" and so on, in the library's errors, warnings and plot labels, the
  tool descriptions an agent reads, the browser app's text and the documentation. A test that matches a message
  with a British spelling (`match="not recognised"`, `match="does not look centred"`) needs the American one.
  Names that a saved study, a workbook or a result holds are unchanged: result keys such as
  `"standardised_coefficient"`, the reference catalog's data and module path
  (`process_improve.experiments.reference_catalogue`), and the settings key `colour_map` of a study.
