- **`import process_improve` and `import process_improve.experiments` no longer load scikit-learn
  and statsmodels.** Each public name is imported from its module when it is first asked for
  (PEP 562), so `from process_improve.experiments import generate_design` loads the design
  generators and nothing a model is fitted with. Every name is importable as before, from the same
  places, and the subpackages are still attributes of the package (`process_improve.univariate`).
  One thing shows: where statsmodels or scikit-learn is missing or broken, the error now comes
  when a name that needs it is first used, no longer at `import process_improve`. (#1235)
