- **`batch.ResampleAligner`, `batch.DTWAligner` and `batch.BatchFeatureExtractor`: an alignment, a feature
  table and a batch model in one scikit-learn `Pipeline`.** `ResampleAligner` resamples every batch onto
  the number of samples of a reference batch, as `resample_to_reference` does, and keeps that length for
  new batches. `DTWAligner` runs `batch_dtw` in `fit` and keeps its reference trajectory and tag weights,
  so `transform` aligns a new batch as the training batches were aligned: a training batch comes back
  exactly as `batch_dtw` returned it. Both choose the most typical batch as the reference when none is
  named, and refuse a tag with missing values, naming the batch and the tag. `BatchFeatureExtractor`
  turns the batches into one row per batch of features from `batch.features` (mean, standard deviation,
  minimum, maximum, last value, and others), one column per tag and feature, as the X block of a
  `PLS` model of the final quality. The three were in the closed #461. (#1105)
- **`batch_dtw` also returns `reference_trajectory`**, the trajectory its last iteration aligned every
  batch to, in the original units. With the last row of `weight_history` it aligns a new batch as the
  batches given were aligned. (#1105)
