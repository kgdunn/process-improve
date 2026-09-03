# SBR batch PCA legacy cross-check fixture

Reference numerical fixture for checking a batchwise-unfolded
`process_improve.PCA` on the simulated styrene-butadiene rubber (SBR) batch
reactor data against the output of the legacy MATLAB latent-variable toolkit
that the 2011-2012 short course used. The same data drive the SBR case study
in `docs/user_guide/case_studies/batch/`.

## Files

| File | Role |
| --- | --- |
| `prepare_fixture.py` | Reads the two archived `.mat` files, writes the three reference files below, and writes the `sbr-batch-reactor.xlsx` workbook that is hosted on openmv.net. Needs `scipy` and `openpyxl`; it is not run in CI. |
| `reference_scores.csv` | Scores T (53 x 2) of the legacy batch PCA: `batch_id, t1, t2`. |
| `reference_loadings.csv` | Loadings P (1800 x 2) of the legacy batch PCA, labelled `tag, sequence, p1, p2`. |
| `reference_metadata.json` | Provenance (repository, commit, blob SHAs), the name mapping, the legacy model settings, and the R2 values the legacy unit test asserts. |

The pytest cross-check (`tests/batch/test_sbr_reference.py`) downloads the
data from openmv.net and is skipped when the download fails, so an offline
run does not break CI.

## Regenerating the fixture

```bash
uv run python tests/fixtures/sbr_batch_pca/prepare_fixture.py \
    --sbrdata /path/to/SBRDATA.mat \
    --expected /path/to/SBR-expected.mat \
    --workbook /path/to/sbr-batch-reactor.xlsx
```

The `.mat` inputs are `tests/SBRDATA.mat` and `tests/SBR-expected.mat` in
the archived [kgdunn/gsk-teaching](https://github.com/kgdunn/gsk-teaching)
repository at commit `87cfb3bbb02fa623bd19bd20e2813bff4a0cf5d4`. They are
not vendored here; the workbook the script writes is the copy served at
`https://openmv.net/file/sbr-batch-reactor.xlsx`.

## What the legacy numbers are

`unit_tests.m::PCA_batch_data` in that repository builds one batch block
from all nine trajectories (53 batches of 200 samples), fits
`lvm({'X', batch_X}, min_lv = 2)`, and asserts the per-component R2 of
`[0.17085, 0.100531]` and the stored scores and loadings. Its `assertEAE`
comparison keeps two significant figures and ignores the sign of every
column, because a principal component is defined up to its sign.

Two details of the MATLAB layout matter when comparing:

- MATLAB unfolds time-major: all nine tags at sample 0, then all nine tags
  at sample 1, and so on. Row `k * 9 + j` of the loadings is tag `j` at
  sample `k`. `reference_loadings.csv` carries that labelling explicitly so
  the comparison can reindex onto `process_improve`'s tag-major
  `(tag, sequence)` column order.
- The two feed-flow trajectories are constant in the simulation. Their
  unfolded columns have zero variance, so the legacy loadings for the
  `(StyreneFlow, 0)` and `(ButadieneFlow, 0)` cells are exactly zero, and
  `MCUVScaler` maps those columns to exactly zero as well.

## Source data attribution

The trajectories were simulated from a first-principles model of the SBR
reactor for Paul Nomikos, *Statistical process control of batch processes*,
PhD thesis, McMaster University, 1995, and were used as course material by
ConnectMV (Kevin Dunn) in 2011-2012. Batches 34 and 37 carry the same
injected fault (30% more organic impurity in the butadiene feed), starting
at the beginning of batch 37 and midway through batch 34. The reference
numbers are the mathematical output of running the legacy toolkit on those
data; no MATLAB source code is copied into this repository.
