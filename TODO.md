# TODO

> Unified follow-ups for upcoming sessions. Replaces the previous free-form `TODO.txt`
> (last on disk at commit `50815c844c65bc1d6523afb3b2cb35bc6c268789`; recover with
> `git show 50815c8:TODO.txt` if needed).
>
> Sources merged here:
> - Original `TODO.txt` items (free-form ideas accumulated over time).
> - "Deferred / out of scope", "Stubs (planned for later)", and "Manual follow-ups"
>   sections from every Claude-co-authored PR in this repo (#46 – #124).
> - Every in-source `# TODO` / `# FIXME` comment and `raise NotImplementedError`
>   stub found in `process_improve/**/*.py` and `tests/**/*.py`.
>
> Provenance tags on each item:
> - `(#NNN)` — derived from PR description.
> - `(TODO.txt)` — from the original free-form file.
> - `(file.py:line)` — from an in-source TODO/FIXME or stub.

## Multivariate (PCA / PLS / TPLS / MBPCA / MBPLS)

- [ ] Add plotting routines on MV models: `model.plot_screeplot(...)` etc.; plots accept a `pc_depth` argument so a 3D plot is made instead. (TODO.txt)
- [ ] PLS lacks an SPE limit — add it. (TODO.txt)
- [ ] Fill in `R2*` attributes on MV models. (TODO.txt)
- [ ] `print(model)` raises `'PCA_missing_values' object has no attribute 'copy'` — fix. (TODO.txt)
- [ ] Implement sklearn-style `fit_transform` on `PCA`. (TODO.txt)
- [ ] Add a post-filter helper: snap values within ±eps of zero to zero. (TODO.txt)
- [ ] Add VIP for PCA, PLS, TPLS. (TODO.txt, #54 — unmerged)
- [ ] Prediction intervals for PLS. (TODO.txt)
- [ ] Jackknife confidence intervals for PLS coefficients. (TODO.txt)
- [ ] Contribution plots and contribution-plot calculation. (TODO.txt)
- [ ] In NIPALS, use the column of X with the greatest variance as the starting score after iteration 1. (TODO.txt)
- [ ] PLS with missing values via TSR methods (Folch-Fortuny — see Reference snippets). (TODO.txt)
- [ ] Investigate Robust PLS (Serneels/Croux/Filzmoser/Van Espen 2005, Partial Robust M-regression). (TODO.txt)
- [ ] Port the legacy `calc_limits` MATLAB block (verbatim in *Reference snippets*) into Python. (TODO.txt)
- [ ] Verify sklearn pipeline compatibility (clone, `get_params` / `set_params`) for refactored PLS. (#59)
- [ ] Verify numerical equivalence of refactored PLS with sklearn `PLSRegression` on reference datasets. (#59)
- [ ] Add larger multi-block reference datasets (FMC, SBR, DuPont) once `.mat` conversion is wanted. (#124)
- [ ] Multi-block batch-wise unfolding — extend `process_improve/batch/preprocessing.py`. (#124)
- [ ] Multi-block cross-validation helper (single-block `Resampler` exists; multi-block CV needs its own pass). (#124)
- [ ] Optional thin `Block` wrapper class around `dict[str, DataFrame]` (only if usage warrants). (#124)
- [ ] Implement TSR for PLS. (multivariate/methods.py:1033)
- [ ] Implement PMP for PLS. (multivariate/methods.py:1035)
- [ ] Provide a scaling vector (TPLS-related). (multivariate/methods.py:1698)
- [ ] Complete the unfinished code block. (multivariate/methods.py:1961)
- [ ] Check column-name consistency between fit and predict for new data. (multivariate/methods.py:2415)
- [ ] Address the open `TODO:` block in TPLS. (multivariate/methods.py:2660)
- [ ] Re-derive `feature_importance["D"]` against deflated matrices `S(V^TS)^{-1}`. (multivariate/methods.py:2922)
- [ ] Re-derive `feature_importance["F"]` against deflated matrices `P(_^TP)^{-1}`. (multivariate/methods.py:2923)
- [ ] Add missing-data support to MBPLS (currently raises `NotImplementedError`). (multivariate/methods.py:3405)
- [ ] Add missing-data support to MBPCA (currently raises `NotImplementedError`). (multivariate/methods.py:4138)
- [ ] Wire the `show_labels` flag through plot helpers. (multivariate/plots.py:103, multivariate/plots.py:629)
- [ ] Decide what to plot when `with_a` is zero or > A. (multivariate/plots.py:619)
- [ ] Constrain `conf_level` to < 1. (multivariate/plots.py:623)

## Batch

- [ ] Percentage-scale x-axis for alignment, with configurable resolution. (TODO.txt)
- [ ] Missing-data handling in batch — see verbatim `fill_na_values` in *Reference snippets*. (TODO.txt)
- [ ] Smoothing tools: lowess and Savitzky–Golay filter for batch data. (TODO.txt)
- [ ] Add tests for `f_cross` and `f_elbow`. (TODO.txt)
- [ ] Don't force batch dict keys to `str`; investigate `align_with_path` adding `batch_id` as a column and crashing on `synced.iloc[row, :] = np.nanmean(temp, axis=0)`. (TODO.txt)
- [ ] Implement `f_robust_mad` — currently raises `AssertionError("This next line of code fails. Fix it.")`. (#89, batch/features.py:295)
- [ ] Resolve outstanding `# TODO: check this out still`. (batch/features.py:372)
- [ ] Investigate change-point detection via `ruptures` (https://github.com/deepcharles/ruptures). (batch/features.py:403)
- [ ] Address the bare `# TODO` markers on two feature functions. (batch/features.py:480, batch/features.py:490)
- [ ] Refactor to scikit-learn-style `.fit()` / `.apply()` API. (batch/preprocessing.py:42)
- [ ] Consider including `f_iqr` as a feature. (batch/preprocessing.py:57)
- [ ] Handle the `DataFrame` input cases. (batch/preprocessing.py:102, batch/preprocessing.py:125)
- [ ] Revisit thesis page 181: handle multiple points in the target. (batch/preprocessing.py:174)
- [ ] Document the function completely. (batch/preprocessing.py:308)
- [ ] Try sum-of-absolute-values weighting alongside quadratic. (batch/preprocessing.py:367)
- [ ] Allow leaving out worst batches when computing weights. (batch/preprocessing.py:370)
- [ ] Make the weighting choice a configurable setting. (batch/preprocessing.py:375)
- [ ] Implement `group_by_batch` with hierarchical column indexing. (batch/data_input.py:128, batch/data_input.py:134, batch/data_input.py:165)
- [ ] Allow user-specified `band` for alignment. (batch/alignment_helpers.py:10)
- [ ] Add Sakoe-Chiba constraints. (batch/alignment_helpers.py:22)
- [ ] Replace pre-3.9 dataclass workaround with pydantic. (batch/plotting.py:326, batch/plotting.py:327)
- [ ] Fix the `[[None]]` placeholder that does not work. (batch/plotting.py:570)
