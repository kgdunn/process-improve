# R reference environment

Some analyses in this package have a long-established R implementation: lme4, lmerTest
and pbkrtest for mixed models, SensMixed for the mixed assessor model of sensory panels.
This Docker image runs them, so that tests can pin values computed in R and anyone can
recompute them. R is not needed to run the test suite; only to regenerate a fixture.

## Targets

| Target | R | Packages | Use |
| --- | --- | --- | --- |
| `mixed` | 4.4.2 | lme4 1.1-36, lmerTest 3.1-3, pbkrtest 0.5-3, jsonlite | REML fits, Satterthwaite and Kenward-Roger tests |
| `sensmixed` | 3.4.4 | SensMixed 2.1-0, lmerTest 2.0-36, lme4 1.1-15, jsonlite | SensMixed's mixed assessor model |

Each target installs from a dated snapshot of CRAN, so a rebuild gets the same versions.

SensMixed 2.1-0 (March 2018) was archived from CRAN in 2020 and no longer runs on a
current stack: R 4.3 made its `&&` on vectors an error, and its random-effect tables
break with later lme4. The `sensmixed` target therefore rebuilds the environment it was
released for, R 3.4.4 with CRAN as of 2018-04-01, compiled from source (about 5
minutes). In that R, `Rscript -e` takes a single line; put longer code in a script.

## Building

```bash
docker build --target mixed -t process-improve-r:mixed tools/r
docker build --target sensmixed -t process-improve-r:sensmixed tools/r
```

Behind a proxy, pass it and, if it intercepts TLS, its CA bundle. With the proxy on the
host's loopback interface, the build also needs the host's network:

```bash
docker build --network host --build-arg HTTPS_PROXY --build-arg NO_PROXY \
    --secret id=ca_bundle,src=/path/to/ca-bundle.crt \
    --target mixed -t process-improve-r:mixed tools/r
```

## Running a script

Mount the repository at `/w`, the working directory, and run from there:

```bash
docker run --rm -v "$PWD:/w" process-improve-r:mixed Rscript tests/fixtures/split_plot_lmer/reference.R
```

## Fixtures computed here

- `tests/fixtures/split_plot_lmer`: the split-plot REML analysis
  (`analyze_experiment(..., analysis_type="split_plot")`) against lme4, lmerTest and
  pbkrtest.
