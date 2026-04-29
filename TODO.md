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
> - `(#NNN)` - derived from PR description.
> - `(TODO.txt)` - from the original free-form file.
> - `(file.py:line)` - from an in-source TODO/FIXME or stub.

## Multivariate (PCA / PLS / TPLS / MBPCA / MBPLS)

- [ ] Add plotting routines on MV models: `model.plot_screeplot(...)` etc.; plots accept a `pc_depth` argument so a 3D plot is made instead. (TODO.txt)
- [ ] Fill in `R2*` attributes on MV models. (TODO.txt)
- [ ] `print(model)` raises `'PCA_missing_values' object has no attribute 'copy'` - fix. (TODO.txt)
- [ ] Add a post-filter helper: snap values within ±eps of zero to zero. (TODO.txt)
- [ ] Add VIP for PCA, PLS, TPLS. (TODO.txt, #54 - unmerged)
- [ ] Prediction intervals for PLS. (TODO.txt)
- [ ] Jackknife confidence intervals for PLS coefficients. (TODO.txt)
- [ ] Contribution plots and contribution-plot calculation. (TODO.txt)
- [ ] In NIPALS, use the column of X with the greatest variance as the starting score after iteration 1. (TODO.txt)
- [ ] PLS with missing values via TSR methods (Folch-Fortuny - see Reference snippets). (TODO.txt)
- [ ] Investigate Robust PLS (Serneels/Croux/Filzmoser/Van Espen 2005, Partial Robust M-regression). (TODO.txt)
- [ ] Port the legacy `calc_limits` MATLAB block (verbatim in *Reference snippets*) into Python. (TODO.txt)
- [ ] Verify sklearn pipeline compatibility (clone, `get_params` / `set_params`) for refactored PLS. (#59)
- [ ] Verify numerical equivalence of refactored PLS with sklearn `PLSRegression` on reference datasets. (#59)
- [ ] Add larger multi-block reference datasets (FMC, SBR, DuPont) once `.mat` conversion is wanted. (#124)
- [ ] Multi-block batch-wise unfolding - extend `process_improve/batch/preprocessing.py`. (#124)
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

## Batch

- [ ] Percentage-scale x-axis for alignment, with configurable resolution. (TODO.txt)
- [ ] Missing-data handling in batch - see verbatim `fill_na_values` in *Reference snippets*. (TODO.txt)
- [ ] Smoothing tools: lowess and Savitzky–Golay filter for batch data. (TODO.txt)
- [ ] Add tests for `f_cross` and `f_elbow`. (TODO.txt)
- [ ] Don't force batch dict keys to `str`; investigate `align_with_path` adding `batch_id` as a column and crashing on `synced.iloc[row, :] = np.nanmean(temp, axis=0)`. (TODO.txt)
- [ ] Implement `f_robust_mad` - currently raises `AssertionError("This next line of code fails. Fix it.")`. (#89, batch/features.py:295)
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

## Univariate

- [ ] Distribution-fit check (NIST handbook §3.5.7 - see *Reference snippets*). (TODO.txt)
- [ ] Outlier detection: integrate / wrap `pyod`. (TODO.txt)
- [ ] Grubbs / Tietjen–Moore single & multiple outlier tests (NIST §3.5.h, §3.5.h3 - see *Reference snippets*). (TODO.txt)
- [ ] Holm post-hoc multiple-comparisons test. (TODO.txt)
- [ ] Robust scale - port the verbatim Mosteller–Tukey MATLAB function in *Reference snippets*. (TODO.txt)
- [ ] Follow up on the `rips-irsp.com/article/10.5334/irsp.82/` reference (likely a multiple-comparisons or post-hoc test to add). (univariate/metrics.py:419)

## Monitoring

- [ ] Return RSD as well: `rsd = (spread / center) * 100`. (monitoring/metrics.py:67)

## Bivariate

- [ ] Robustify the elbow-point detection. (bivariate/methods.py:125)

## Regression

- [ ] Prediction-interval function for linear regression that accepts any `x`. (TODO.txt)

## Experiments / DOE

- [ ] Build out the product-development notebook - full outline preserved verbatim under *Reference snippets*. (TODO.txt)
- [ ] Implement `ridge_analysis` (trace optimum along increasing radii). Currently a stub; removed from the `optimize_responses` tool enum. (#74, #79)
- [ ] Implement `pareto_front` (multi-objective NSGA-II). Currently a stub; removed from the `optimize_responses` tool enum. (#74, #79)
- [ ] Verify `execute_tool_call("analyze_experiment", ...)` works end-to-end. (#72)
- [ ] Test `analyze_experiment` with real DOE datasets (LDPE, etc.). (#72)
- [ ] Handle inputs whose shape is ≥ 2 columns (category branch). (experiments/structures.py:368)
- [ ] Implement the `pd.DataFrame` branch (currently `raise NotImplementedError("Handle this case still")`). (experiments/structures.py:388)
- [ ] Check that all indexes are common before merging, or use pandas's same-index merge functionality. (experiments/structures.py:390)
- [ ] Replace the unhandled-type `raise NotImplementedError` in the number-formatting fallback. (experiments/models.py:30)

## Visualization

- [ ] Implement `raincloud` - currently a stub that returns `None`. (#89, visualization/plots.py:4)

## Tests

- [ ] Flesh out partial / placeholder test bodies - R cross-checks, SPE-vs-Simca-P comparison, Plotly assertions, etc. (tests/test_multivariate.py:388, tests/test_multivariate.py:413, tests/test_multivariate.py:478, tests/test_multivariate.py:703, tests/test_multivariate.py:747, tests/test_multivariate.py:1012, tests/test_multivariate.py:1226, tests/test_multivariate.py:1246, tests/test_multivariate.py:1309, tests/test_multivariate.py:2073, tests/test_multivariate.py:2220)
- [ ] Investigate the sequence that yields Sn = 0 despite variability. (tests/test_univariate.py:113)
- [ ] Complete the robust-case test. (tests/test_univariate.py:394)
- [ ] Address the bare `# TODO`. (tests/test_univariate.py:843)
- [ ] Add NaN-at-start example. (tests/test_monitoring.py:116)
- [ ] Add NaN-in-warm-up example. (tests/test_monitoring.py:117)
- [ ] Migrate from `unittest` to `pytest`. (tests/test_doe.py:194)
- [ ] Switch to `pathlib.Path`. (tests/test_doe.py:195)
- [ ] Add pandas-Series input tests. (tests/test_regression.py:19)
- [ ] Handle NaN in vectors. (tests/test_regression.py:20)
- [ ] Add shape edge cases (5×2 / 7×1, 5×4 / 5×1, 5×5+intercept failure). (tests/test_regression.py:112, tests/test_regression.py:113, tests/test_regression.py:114)
- [ ] Test the `age` column. (tests/batch/test_features.py:73)
- [ ] Tighten the test depending on how DTW termination is finalised. (tests/batch/test_preprocessing.py:215)

## Documentation

- [ ] Add documentation (general carry-over). (TODO.txt)
- [ ] Update Quick Start link `https://kgdunn.github.io/process-improve/quickstart.html` if deployed docs path changes. (#119)

## CI / release / infra

- [ ] Confirm Codecov "Missing Base Commit" is gone after the next PR. (#117)
- [ ] Confirm `CODECOV_TOKEN` is set under repo Settings → Secrets and variables → Actions. (#118)
- [ ] After merge, watch first `Unit tests` run on `main` - codecov upload should run on `ubuntu-latest` + `3.13` only. (#118)
- [ ] Confirm README codecov badge resolves to a percentage instead of "unknown". (#118)
- [ ] Update the README CI badge URL when `run-tests.yml` is renamed/replaced. (#119)
- [ ] Settings → Code security → enable Dependabot alerts + Dependabot security updates. (#97)
- [ ] Settings → Code security → confirm secret scanning is enabled. (#97)
- [ ] Bump version in `pyproject.toml` on `main` and confirm a tag + release appear on the releases page. (#87)
- [ ] Ensure PyPI trusted publisher is configured for this repo (one-time manual step via PyPI browser UI). (#86)

## External / downstream coordination

- [ ] After PyPI publish: downstream `factorial` PR bumps `process-improve` pin to `>=1.6.0` and drops the `requires_simulator_tools` skip guard. (#103)
- [ ] Downstream `factorial` reproducible-export service: read `spec["rng"]["seed_param"]` instead of the hardcoded `_DESIGN_SEED_KEYS` set. (#105)
- [ ] Companion PR in `kgdunn/agentic-doe` wires the hosted backend through `safe_execute_tool_call`. (#91)
- [ ] Land the unmerged tools-architecture restructure (PR #55 still open). (#55)

## Reference snippets

Preserved verbatim from the original `TODO.txt` so the context referenced by the
checkboxes above is not lost.

### `fill_na_values` - crude batch missing-value fill

```python
# Fill in missing values
def fill_na_values(df):
    "Very very crude method for now"
    return df.fillna(method='bfill').fillna(method='ffill')

missing_filled = {}
for batch_id, batch in  df_dict.items():
    missing_filled[batch_id] = fill_na_values(batch)
```

### Elbow-method reference

- https://github.com/Mathemilda/Numeric_ElbowMethod_For_K-means/blob/master/EstimatedClusterNumberWithWCSS.py

### Univariate references

- Distribution check: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35g.htm
- Outlier detection: https://pyod.readthedocs.io/en/latest/
- Grubbs / Tietjen-Moore: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h1.htm
- Multiple outliers: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm
- https://www.itl.nist.gov/div898//software/dataplot/refman1/auxillar/grubtest.htm
- https://www.itl.nist.gov/div898//software/dataplot/refman1/auxillar/tietjen.htm

### Holm post-hoc test reference

- https://stats.libretexts.org/Bookshelves/Applied_Statistics/Book%3A_Learning_Statistics_with_R_-_A_tutorial_for_Psychology_Students_and_other_Beginners_(Navarro)/14%3A_Comparing_Several_Means_(One-way_ANOVA)/14.06%3A_Multiple_Comparisons_and_Post_Hoc_Tests

### Multivariate references

- PLS with TSR methods: https://riunet.upv.es/bitstream/id/303213/PCA%20model%20building%20with%20missing%20data%20new%20proposals%20and%20a%20comparative%20study%20-%20Folch-Fortuny.pdf
- Robust PLS: S. Serneels, C. Croux, P. Filzmoser, P.J. Van Espen, *Partial Robust M-regression*, Chemometrics and Intelligent Laboratory Systems, 79 (2005), 55-64.

### `robust_scale` - Mosteller & Tukey, MATLAB

```matlab
function  v = robust_scale(a)
% Using the formula from Mosteller and Tukey, Data Analysis and Regression,
% p 207-208, 1977.
    n = numel(a);
    location = median(a);
    spread_MAD = median(abs(a-location));
    ui = (a - location)/(6*spread_MAD);
    % Valid u_i values used in the summation:
    vu = ui.^2 <= 1;
    num = (a(vu)-location).^2 .* (1-ui(vu).^2).^4;
    den = (1-ui(vu).^2) .* (1-5*ui(vu).^2);
    v = n * sum(num) / (sum(den))^2;
end
```

### `calc_limits` - legacy MATLAB / Python hybrid block to port

```python
def calc_limits(self):
    """
    Calculate the limits for the latent variable model.

    References
    ----------
    [1]  SPE limits: Nomikos and MacGregor, Multivariate SPC Charts for
            Monitoring Batch Processes. Technometrics, 37, 41-59, 1995.

    [2]  T2 limits: Johnstone and Wischern?
    [3]  Score limits: two methods

            A: Assume that scores are from a two-sided t-distribution with N-1
            degrees of freedom.  Based on the central limit theorem

            B: (t_a/s_a)^2 ~ F_alpha(1, N-1) distribution if scores are
            assumed to be normally distributed, and s_a is chi-squared
            variable with N-1 DOF.

            critical F = scipy.stats.f.ppf(0.95, 1, N-1)
            which happens to be equal to (scipy.stats.t.ppf(0.975, N-1))^2,
            as expected.  Therefore the alpha limit for t_a is equal to
            np.sqrt(scipy.stats.f.ppf(0.95, 1, N-1)) * S[:,a]

            Both methods give the same limits. In fact, some previous code was:
            t_ppf_95 = scipy.stats.t.ppf(0.975, N-1)
            S[:,a] = np.std(this_lv, ddof=0, axis=0)
            lim.t['95.0'][a, :] = t_ppf_95 * S[:,a]

            which assumes the scores were t-distributed.  In fact, the scores
            are not t-distributed, only the (score_a/s_a) is t-distributed, and
            the scores are NORMALLY distributed.
            S[:,a] = np.std(this_lv, ddof=0, axis=0)
            lim.t['95.0'][a, :] = n_ppf_95 * S[:,a]
            lim.t['99.0'][a, :] = n_ppf_99 * [:,a]

            From the CLT: we divide by N, not N-1, but stddev is calculated
            with the N-1 divisor.
    """
    for block in self.blocks:
        N = block.N
        # SPE limits using Nomikos and MacGregor approximation
        for a in xrange(block.A):
            SPE_values = block.stats.SPE[:, a]
            var_SPE = np.var(SPE_values, ddof=1)
            avg_SPE = np.mean(SPE_values)
            chi2_mult = var_SPE / (2.0 * avg_SPE)
            chi2_DOF = (2.0 * avg_SPE ** 2) / var_SPE
            for siglevel_str in block.lim.SPE.keys():
                siglevel = float(siglevel_str) / 100
                block.lim.SPE[siglevel_str][:, a] = chi2_mult * stats.chi2.ppf(
                    siglevel, chi2_DOF
                )

            # For batch blocks: calculate instantaneous SPE using a window
            # of width = 2w+1 (default value for w=2).
            # This allows for (2w+1)*N observations to be used to calculate
            # the SPE limit, instead of just the usual N observations.
            #
            # Also for batch systems:
            # low values of chi2_DOF: large variability of only a few variables
            # high values: more stable periods: all k's contribute

            for siglevel_str in block.lim.T2.keys():
                siglevel = float(siglevel_str) / 100
                mult = (a + 1) * (N - 1) * (N + 1) / (N * (N - (a + 1)))
                limit = stats.f.ppf(siglevel, a + 1, N - (a + 1))
                block.lim.T2[siglevel_str][:, a] = mult * limit

            for siglevel_str in block.lim.t.keys():
                alpha = (1 - float(siglevel_str) / 100.0) / 2.0
                n_ppf = stats.norm.ppf(1 - alpha)
                block.lim.t[siglevel_str][:, a] = n_ppf * block.S[a]
```

### Product-development notebook narrative

Verbatim from `TODO.txt` - captures the storyboard for the planned PD notebook
that walks through latent-variable model inversion for new-product design.

```
Output
3 dofs available in general
1 dof
2 dof.

PCA space. For outputs. On 7 properties A to G.
Shows the correlations and quickly seeing gaps

PCA on processing settings and blend properties. RX. Related to the scores. Often a 1 to 1 relationship of the outputs to the scores. So this is valid.

Use the 5 properties of A to E from foods data set. Show that constraints are also possible. Indicates multiple solutions. Also mention the nulll space. Of multiple solutions. T -> X.

Then go to the cinac case study. Show that the lvs are interesting and have meaning. We can design new product here. Y -> T.

Now we can also go directly from Y to X, with a NLP solution. But it goes via the scores T. And it can handle additional constraints.
Would be cool to demo this.

Lastly how do you start out?
* dB is partially available.
* R selected from a regular mixture DoE.
* Calculate XR. Do a doe in this space to select a subset of experimental conditions.
* run those expts.
* Start up your mixture model
* See where there are holes literally in the spaces.
* Use LV model inversion to find which process settings and ratios and materials are needed to fill in those gaps
* Targeted DoE.


raw material properties affect what final product properties is made clear in the pls loadings plot. Again, something that is not seen in NN model.

Build a model explorer for the pls model of muteii for the rubbers, showing how the blend properties correlated with the 8 outputs.

Show the approach on p 23 of muteii paper for how to start out. You can use what you have and add to it via a d optimal doe.
Show figure 4 and emph that the full 111 expats were not needed. Only 17.
```
