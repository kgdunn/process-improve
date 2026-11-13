- **`import process_improve` and `import process_improve.experiments` no longer load scikit-learn
  and statsmodels.** Each public name is imported from its module when it is first asked for
  (PEP 562), so `from process_improve.experiments import generate_design` loads the design
  generators and nothing a model is fitted with. Every name is importable as before, from the same
  places. The subpackages and modules are still attributes of their package
  (`process_improve.univariate`, `experiments.models.lm`), and several threads may ask a package
  for its names at once. Three things show:
  - Where statsmodels or scikit-learn is missing or broken, the error comes when a name that needs
    it is first used, no longer at `import process_improve`.
  - `import process_improve` alone no longer imports plotly, so it no longer registers the Plotly
    themes (`pi_journal` and the others) as a side effect. They are registered when
    `process_improve.visualization` is imported, as documented, and before the library draws a
    figure.
  - The packages scikit-learn, statsmodels and plotly are now imported when first used, which may
    be in a thread. A program that imports modules of `process_improve` in one thread while
    another first uses the library should import them before the threads start, as for any
    package whose dependencies import each other in a circle. (#1235)
