- **A multivariate page in the browser app, at `mva/`.** PCA and PLS on the LDPE example or on your own CSV or
  Excel file, with Python running in the page (Pyodide): a canvas data table whose column headers show each
  variable's distribution, run order and missing values, with the selected rows drawn over them; linked score,
  loading, SPE and T2 plots, where a lasso in one selects the rows everywhere; rows excluded from the model but
  still drawn, hollow, where the refitted model projects them; "Why are these rows different?", which ranks the
  variables behind a group's shift with their raw data beside the bars; recipe steps that write the Python that
  repeats them; and project files that refit and check themselves when opened. The page shares the app's wheel
  with the designed-experiments page.
