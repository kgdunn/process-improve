- **`batch_dtw` gives a tag that does not vary within any batch zero weight, with a warning naming
  it.** Such a tag took all of the weight: with two tags that vary and one that does not, the
  weights were `[0, 0, 3]`, and the convergence test then followed only the flat tag. A tag that
  holds one value within each batch, but a different one between batches, took a share of the
  weight too. Neither says anything about timing. The tags that vary now share the weight as they
  would without the flat one, so naming it changes nothing in the alignment. A call in which no
  named tag varies raises `NotEnoughVarianceError`, a `ValueError`. (#1184)
- **`batch_dtw`'s `last_average_batch` keeps a named tag that does not vary**, as the aligned
  batches do; it dropped it without a word. (#1184)
- **`resample_to_reference` refuses a tag with a missing value**, naming each batch and tag, and
  points at `fill_gaps`. One missing value made the whole resampled tag NaN under the default cubic
  interpolation. `batch_dtw` refuses a missing value in the same way. `find_reference_batch`, which
  resamples first, gives the same error. (#1184)
- **`find_reference_batch` works for any number of batches.** With 1 to 4 batches and
  `judged_rows="fitted"` it raised "Found array with 1 sample(s)", as it did with 1 batch under any
  setting, and with 5 it warned that more components were asked for than the data allow. Its PCA fitted `n - 1` components to `n` batches, which passes through every batch, so
  every batch had the same T² and rounding chose the reference. It now fits at most `n - 2`, so with
  3 to 5 batches, or when the T² screen keeps 5 or fewer, the reference can differ from the one
  given before. One batch is its own reference; of two, the one of average length comes first.
  (#1184)
- **`f_agemin` and `f_agemax` give NaN for a batch in which a tag has no observed value**, as
  `f_min` and `f_max` do. Under pandas 3 they raised `ValueError`; pandas 2 gave NaN. (#1184)
- **The `BatchPCA` example runs.** It passed `reference_batch="1"` to batches keyed by integers,
  which raised `KeyError`. (#1184)
