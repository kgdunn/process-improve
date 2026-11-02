- **`PLS.fit` labels the rows of a 1-D response array by X's row labels, as it does a 2-D one.**
  A 1-D NumPy `y`, the shape sklearn pipelines pass, gave `predictions_` and `y_scores_` rows
  numbered 0 to N-1 even when X was a DataFrame with labelled rows, while `scores_` and
  `hotellings_t2_` carried X's labels. A `pd.Series` response keeps its own labels, as before.
  (#1027)
