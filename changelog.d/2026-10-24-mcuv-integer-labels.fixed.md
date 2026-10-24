- **`MCUVScaler` keeps integer column labels.** It took its labels from sklearn's `feature_names_in_`,
  which sklearn sets only for string labels, so a frame with integer labels (a spectrum's wavelengths, or
  a CSV read with `header=None`) came out labelled 0..K-1. The loadings of a model fitted to the scaled
  frame were then labelled with the wrong wavelengths. `center_`, `scale_`, and the output of `transform`
  and `inverse_transform` now carry the frame's own labels; ndarray input is still numbered from 0.
