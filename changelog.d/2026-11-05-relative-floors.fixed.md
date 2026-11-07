- **The multivariate family gives the same model for the same data in any units.** A floor that
  decides "no spread" or "zero" is now relative to the data's own scale, where it was an absolute
  1.5e-8 (or 1.5e-154) in the data's units. On LDPE in units 1e9 times smaller, with nothing else
  changed:
  - Hotelling's T² left out a component as having no spread when its score standard deviation
    was below 1.5e-8 while every one was below 1: `PCA`'s T² counted one component of four, and
    `PLS(scale=False)`'s was 0 for every row. A component now has no spread only at 1.5e-8 times
    the largest. A model whose largest score sd is 1 or more keeps its T²; one whose component had
    a score sd between 1.5e-8 times the largest and 1.5e-8 itself now counts that component.
  - `PCA.select_n_components(scale_inside_folds=False)` recommended 4 components instead of 8, with
    Q² NaN: the reference sum of squares counted as zero below 1.5e-8, and the imputation stopped
    after one pass when the held-out cells were smaller than 1. A column now has no Q² only when
    its observed values are all equal, as `MCUVScaler` reads one, and the imputation stops at a
    change relative to the held-out cells' own size.
  - `score_contributions(..., scaling="maximum")` and `scaling="within"`, single-block and
    multi-block, left bars below 1.5e-8 unscaled. `"maximum"` now divides unless every bar is 0, and
    `"within"` leaves a row alone when its total is at most 1.5e-8 times the largest row's.
  - `PRM(scale=False)` moved its predictions by up to 1.05 standard deviations of y, and its row
    weights by up to 0.51: its L1 median stopped at an absolute 1e-8, and the floors on the median
    residual, the median distance, the score lengths and the stopping threshold were in the data's
    units. They are now relative to each block's robust spread. On LDPE in units of 1, `PRM` gives
    the same numbers, bit for bit, and `PRM(scale=False)`'s predictions move by 2e-8 standard
    deviations of y. (#1114)
  (#1070)
- **Data near 1e160 or 1e-160 no longer lose their scaling.** `MCUVScaler`, and with it `PCA` on
  autoscaled data, `PLS`, `PRM`, `MBPCA` and `MBPLS`, and `TPLS`'s own scaling, squared the data to
  take a standard deviation, which overflowed to `inf` beyond about 1e154 and underflowed to 0
  below about 1e-162, and a standard deviation below 1.5e-154 left a column unscaled. Data in
  units 1e150 times larger raised "no variance left" in `PLS` and `MBPLS`, and gave `TPLS` other
  scores with no warning; data in units 1e160 times smaller gave `PCA` and `MBPCA` other scores,
  and raised in `PLS`, `MBPLS` and `TPLS`. Each column is now divided by a power of two near its
  largest value first, which is exact, so ordinary data give the same standard deviation, bit for
  bit. A column counts as unscaled only below the smallest normal float, 2.2e-308. (#1070)
- **`PRM` gives a row far enough out that its squares overflow, beyond about 1e154, a weight of 0.**
  It raised "distances must be finite". (#1114)
