# Split-plot analysis against lme4, lmerTest and pbkrtest

Reference values for `TestAgainstLmerTest` in `tests/test_experiments_split_plot.py`,
which checks `analyze_experiment(..., analysis_type="split_plot")` against R.

## Files

| File | Role |
| --- | --- |
| `unbalanced.csv` | An unbalanced split plot: seven whole plots (`plot`) of 2 to 5 runs; `A` changes only between whole plots, `B` within them. |
| `reference.R` | Fits `y ~ A * B + (1 \| plot)` to it, and Box, Hunter and Hunter's corrosion experiment (`src/process_improve/datasets/experiments/corrosion.csv`) with sum-to-zero contrasts, by REML; writes `reference.json`. |
| `reference.json` | lme4's variance components, estimates and standard errors; lmerTest's Satterthwaite and pbkrtest's Kenward-Roger df; the Type III F-tests for the corrosion data. `generated_with` records the R and package versions. |

## What it shows

On the balanced corrosion data, lmerTest, pbkrtest and this package agree on the
variance components, F, df and p. On the unbalanced data the estimates, standard errors
and variance components agree, and this package's df equal pbkrtest's Kenward-Roger df:
they come from the expected information, as Kenward and Roger's do. lmerTest's
Satterthwaite df use the observed information instead, and are a little larger there.

lme4 estimates by numerical optimisation, which agrees with this package's closed-form
fit to about 2e-7; the test allows 1e-5.

## Regenerating

From the repository root, in the `mixed` target of `tools/r/Dockerfile` (see
`tools/r/README.md`):

```bash
docker run --rm -v "$PWD:/w" process-improve-r:mixed Rscript tests/fixtures/split_plot_lmer/reference.R
```
