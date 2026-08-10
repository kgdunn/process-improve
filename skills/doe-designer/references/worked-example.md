# Worked example: screening seven factors

A complete session, start to finish. Every number below came from actually
running these commands; nothing is illustrative.

The situation: a chemist wants to raise the yield of a reaction. They can
think of seven things that might matter and can afford about 40 runs.

## 1. Interview

Established by asking:

- **Response**: Yield, in percent. Maximise. Measurement repeatability about
  1 percentage point.
- **Factors**: Temperature (150-180 C), Time (30-90 min), Catalyst
  (0.5-2.0 %), Pressure (1-3 bar), Stir (200-600 rpm), pH (5-8),
  Solvent (10-30 %).
- **Budget**: 40 runs, about six weeks.
- **Constraints**: none prohibitive.
- **Prior knowledge**: none firm. Temperature is suspected to matter.

Seven factors and no firm prior knowledge means this is a screening problem,
not an optimisation problem. Do not reach for a response surface here.

## 2. Strategy

```bash
python scripts/doe_tool.py call recommend_strategy --input strategy_spec.json
```

Returned a three-stage plan:

| Stage | Design | Runs | Purpose |
|---|---|---|---|
| 1. Screening | Plackett-Burman | 8 | Find the vital few among 7 factors |
| 2. Optimisation | CCD, 3 center points, rotatable | 17 | Quadratic surface on the survivors |
| 3. Confirmation | 3 replicates at the optimum | 3 | Check the prediction holds |

It also returned transition rules worth quoting to the user, because they say
in advance what to do with each outcome: 0-1 significant factors means the
ranges were too narrow (or the measurement system is the problem); 2-5 means
proceed to optimisation; 6+ means split the factors into subgroups; curvature
detected at the center points means augment to a CCD.

We chose to spend a little more than the plan's 8 runs on stage 1, taking a
16-run resolution IV fractional factorial instead of the 8-run
Plackett-Burman. The reason: at 8 runs every main effect is aliased with
two-factor interactions, and this process is one where a
temperature-by-catalyst interaction is chemically plausible. 16 runs is still
well inside budget and buys main effects clear of all two-factor interactions.

## 3. Generate

```bash
python scripts/doe_tool.py call generate_design --input design_spec.json --output design.json
```

with `design_type: "fractional_factorial"`, `resolution: 4`,
`n_center_points: 0`. Returned 16 runs, 7 factors, declared resolution 4, with
both `design_coded` and `design_actual` (real units) plus a randomised
`run_order`.

## 4. Verify

Never skip this.

```bash
$ python scripts/verify_design.py screen_coded.csv --require-resolution 4 --expect-runs 16 --expect-factors 7

screen_coded.csv
  16 runs x 7 factors
  Resolution IV (strength 3)
  Main effects are clear of each other and of two-factor interactions, but some
  pairs of two-factor interactions are aliased with each other. The standard
  screening choice.
  Moment aberration pattern: 3.267, 11.67, 42.47, 157.3, 591.3, 2252, 8666
  First moment short of its lower bound: K_4

PASSED: every requested check is satisfied.
```

The matrix really is resolution IV, and the pattern matches the published
minimum aberration 2^(7-3) design, so this is not merely resolution IV but the
best 16-run 7-factor design there is.

What to tell the user: *you will learn which of the seven factors matter, and
you will not confuse a main effect with an interaction. You will not be able
to tell certain pairs of interactions apart from each other. If an interaction
turns out to matter, a follow-up will be needed to say which one it is.*

## 5. Run sheet

Hand over `design_actual` in real units, ordered by `run_order`, with an empty
Yield column. Stress that the order is deliberate and should not be
rearranged for convenience: it is what protects the results against drift.

## 6. Analyse

```bash
python scripts/doe_tool.py call analyze_experiment --input analysis_spec.json --output analysis.json
```

with `model: "interactions"` and
`analysis_type: ["effects", "lenth_method", "residual_diagnostics"]`.

Lenth's method rather than an F-test, because 16 runs fitting main effects
plus interactions leaves no clean degrees of freedom for error.

Effects, largest first:

```
Temperature                 13.140
Catalyst                     8.782
Temperature:Catalyst         1.773
Time:Stir                    1.773
Pressure:Solvent             1.773
pH                           0.758
Pressure                     0.470
Time                         0.337
```

Lenth's method: PSE 0.198, margin of error 0.444, simultaneous margin 0.852.
Temperature and Catalyst clear both thresholds comfortably. Nothing else does.

## 7. The teaching moment

Look at those three interaction estimates. `Temperature:Catalyst`,
`Time:Stir` and `Pressure:Solvent` are all exactly 1.773.

That is not a coincidence and it is not a bug. It is the resolution IV
aliasing showing up in the data: those three interactions sit in the same
alias chain, so the design cannot distinguish them. The experiment measured
their *sum*, and the analysis has no choice but to report the same number
three times.

This is exactly what the verification step warned about, made concrete. Say
so to the user in those terms, because it is the difference between a report
they can act on and one that quietly misleads them.

Which of the three is real? Effect heredity says the interaction between two
factors that both have large main effects is far more likely than one between
four inert factors, so `Temperature:Catalyst` is the sensible bet. But it is a
bet. Confirming it needs more runs: a fold-over via `augment_design`, or
simply carrying all three candidates into a stage 2 design that separates them.

## 8. Diagnostics

`residual_diagnostics` returned Durbin-Watson 2.50 (no autocorrelation
concern), Breusch-Pagan p = 0.97 (no heteroscedasticity), and adjusted
R-squared 0.9996.

One caution: with a saturated model, every point has identical Cook's distance
(0.517 here) and leverage. That is a property of the design, not a signal
about the data, and the Shapiro-Wilk test on residuals from a saturated fit is
not meaningful either. Do not over-read diagnostics when there are barely any
residual degrees of freedom; the half-normal plot is more informative.

## 9. Plot

```bash
python scripts/render_plot.py --analysis analysis.json --type pareto --output pareto.png
python scripts/render_plot.py --analysis analysis.json --type half_normal --output half_normal.png
```

The half-normal plot is the one to show. Temperature and Catalyst sit far off
the line; everything else falls on it. The three tied interactions appear as a
single visual point, which makes the aliasing self-evident.

## 10. What to report

> Two of the seven factors drive yield. Raising temperature from 150 to 180 C
> increases yield by about 13 percentage points; raising catalyst from 0.5 to
> 2.0 % adds about 9. The other five (time, pressure, stirring, pH, solvent)
> had no detectable effect over the ranges tested, so they can be set for
> convenience or cost.
>
> There is a hint of an interaction of about 1.8 points, most plausibly
> between temperature and catalyst, but this design cannot separate it from
> two other interactions. It is small relative to the main effects and does
> not change the direction of travel.
>
> Sixteen of the forty-run budget are spent. Recommended next step: a central
> composite design on temperature and catalyst alone, which will find the
> optimum settings, resolve the interaction, and detect any curvature. Note
> that both effects point upward at the edge of the range tested, so the
> optimum may lie outside it: consider widening the ranges in stage 2 if that
> is chemically safe.

That last sentence is the sort of thing that comes from reading the numbers
rather than reciting them, and it is usually the most useful line in the
report.

## 11. Stage 2

```bash
python scripts/doe_tool.py call generate_design --input '{
  "factors": [{"name": "Temperature", "low": 165, "high": 200, "units": "C"},
              {"name": "Catalyst", "low": 1.0, "high": 3.0, "units": "pct"}],
  "design_type": "ccd", "n_center_points": 3, "alpha": "rotatable"}'
```

Then analyse with `model: "quadratic"`, add `lack_of_fit` and
`curvature_test`, optimise with `optimize_responses`, and confirm the
predicted optimum with three replicates checked by `confirmation_test`.
