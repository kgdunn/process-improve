- **`gather()` no longer drops an input without a word (#677).** It kept Column, list, Series and
  DataFrame values and skipped anything else: `gather(A=A, x=np.array([...]), t=(5, 6, 7, 8))` came
  back with column A only. A tuple and a one-dimensional NumPy array are now columns, as they are
  entries of `c()`. Any other input (a number, a two-dimensional array, a dict) raises a
  `TypeError` that names its keyword.
