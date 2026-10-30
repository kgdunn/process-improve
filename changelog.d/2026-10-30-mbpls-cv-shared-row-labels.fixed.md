- **`MBPLS.select_n_components` takes rows by position, so rows may share a label.** A fold's Y
  was taken by label, so a label two rows share brought both rows in for each, the fold's Y no
  longer lined up with its blocks, and the call failed with a `ValueError`. Rows are now taken,
  and their held-out predictions gathered, by position, as `PLS.select_n_components` does.
  (#1028)
