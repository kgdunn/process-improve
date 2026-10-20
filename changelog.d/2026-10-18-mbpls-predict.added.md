- **`MBPLS.predict`** returns the predicted Y of new observations as a `DataFrame` on its original
  scale, as `PLS.predict` does, so an `MBPLS` model works where scikit-learn calls `predict`. 2.0.0
  removed a method of the same name that returned the `diagnose` Bunch; this one returns the
  predictions only, so code that read `.predictions` off the old result raises an
  `AttributeError` rather than reading a wrong value.
