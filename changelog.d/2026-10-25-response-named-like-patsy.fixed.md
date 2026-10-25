- **`analyze_experiment` fits a response named `I`, `C` or `Q`.** A column of that name hid patsy's function of
  the same name, so a quadratic model failed with `PatsyError: 'Series' object is not callable` at `I(x1 ** 2)`.
  The response is now fitted under a stand-in name and reported under its own, as a response named after a
  Python keyword already was. Box and Draper's Exercise 7.9 names its responses I, II and III.
