# The mixed assessor model against SensMixed

Reference values for `tests/test_sensory_mam_reml.py`, which checks
`process_improve.sensory.mixed_assessor_model_reml` against the R package SensMixed.

## Files

| File | Role |
| --- | --- |
| `tvbo.csv` | The TVbo panel: 8 assessors score 12 products, 3 TV sets (`TVset`) by 4 pictures (`Picture`), twice (`Repeat`), on 15 attributes. From Bang and Olufsen; distributed with the R packages lmerTest and SensMixed. |
| `reference.R` | Runs SensMixed's mixed assessor model on four versions of the panel, and checks that its own step-by-step replica reproduces `sensmixed()`; writes `reference.json`. |
| `reference.json` | Per case and attribute: the Type I F-tests, the random-term selection, the terms dropped for zero variance, the final variance components, REML criterion and scaling coefficients. `generated_with` records the R and package versions. |

## Cases

| Case | Product | Replicate | Data |
| --- | --- | --- | --- |
| `one_way` | `Product`, the 12 TV set and picture combinations | `Repeat` | all 192 rows |
| `factorial` | `TVset` and `Picture` | `Repeat` | all 192 rows |
| `unbalanced` | `TVset` and `Picture` | `Repeat` | rows 3, 50, 77, 120 and 160 removed |
| `no_replicate` | `TVset` and `Picture` | none | the two replicates averaged |

The scaling coefficient `beta` of each assessor is `1 + g - mean(g)`, with `g` the
coefficients of SensMixed's `Assessor:x.scaling.private` term and 0 for the assessor whose
column lme4 drops as aliased.

## What it shows

The F-tests, Satterthwaite df, variance components and scaling coefficients agree with
SensMixed to about 1e-6 (2e-4 for the df on unbalanced data), for every attribute and case.
In the one-way case the closed-form `beta` of `mixed_assessor_model` equals SensMixed's to
1e-11.

The random-term selection agrees except where a variance is estimated at zero. SensMixed
drops a term whose standard deviation lme4 estimates below 1e-7 before testing anything,
and lme4's search ends near the zero boundary rather than on it, so which terms it drops
depends on where the search stopped: for some attributes it drops the assessor term, for
others it keeps it at zero. This package's estimates are exact, so every boundary term is
dropped, except the assessor and product-by-assessor terms, which SensMixed means to keep
always. Where the two differ, the final models are the same (a zero variance changes
nothing in them), and only the likelihood-ratio tests of terms related to the dropped one
differ.

## Regenerating

From the repository root, in the `sensmixed` target of `tools/r/Dockerfile` (see
`tools/r/README.md`); it takes about four minutes:

```bash
docker run --rm -v "$PWD:/w" process-improve-r:sensmixed Rscript tests/fixtures/sensmixed_mam/reference.R
```
