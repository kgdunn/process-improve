# Analysing experimental results

Read this once results are in hand. The order matters: diagnostics before
conclusions, always.

## The sequence

1. **Look at the raw data first.** Plot the response against run order. A
   trend there means something drifted during the experiment, and no amount of
   modelling fixes it silently. Check for values that are impossible rather
   than merely surprising.
2. **Fit the model.** `analyze_experiment` with `analysis_type` including
   `effects` and `anova`. Start with `model: "interactions"` for a factorial;
   only go to `quadratic` if the design supports it.
3. **Check the residuals before reading any p-value.**
   `residual_diagnostics`. See the traps below.
4. **Test for lack of fit** if the design has center points or replicates.
   Significant lack of fit means the model form is wrong, and the effect
   estimates are not to be trusted regardless of their p-values.
5. **Test for curvature** if there are center points. Significant curvature
   means a two-level design has hit its limit and the next stage needs axial
   points.
6. **Identify the real effects.** With replication, use the ANOVA F-tests.
   Without, use `lenth_method`.
7. **Report effect sizes in response units**, not just significance. "Raising
   temperature by 30 C increases yield by 14 percentage points" is useful;
   "temperature was significant at p < 0.05" is not, on its own.
8. **Plot.** Pareto or half-normal first, then main effects and interactions
   for what survived.

## Unreplicated factorials

A 2^k design run once has zero degrees of freedom for error if you fit every
interaction, so there is nothing to test against and the ANOVA is undefined.
This is the normal case for screening, not a mistake.

Use `lenth_method`. It estimates the noise from the smallest effects on the
assumption that most of them are inert, which is the same effect-sparsity
assumption that justified running a fractional design in the first place.

The half-normal plot is the visual version of the same idea: inert effects
fall on a straight line through the origin, and the ones that matter fall off
it to the right. It is often more persuasive to a user than a p-value, because
they can see the gap.

Do not pool high-order interactions into an error term without saying so. It
is a defensible move when the design is resolution V or better and the
three-factor interactions are genuinely believed negligible, but it is a
judgement call and it should be stated, not slipped in.

## Effect heredity and hierarchy

Two principles that should shape which models you are willing to fit:

- **Hierarchy**: if an interaction `A:B` is in the model, keep `A` and `B` too,
  even if their individual p-values are unimpressive. Models that violate this
  are not invariant to how the factors were coded, which means the conclusions
  depend on an arbitrary choice.
- **Heredity**: an interaction between two factors that both look inert is
  much more likely to be noise than a real effect. Treat such a finding with
  suspicion, and prefer confirming it before acting on it.

`model_selection` respects these. Hand-built formulas may not, so check.

## Residual diagnostics: the traps

Run `residual_diagnostics` every time and actually read it.

- **Funnel shape** in residuals-vs-fitted: the variance grows with the mean.
  Very common for yields, counts, times and concentrations. Fix with a
  transformation, usually log; `box_cox` will suggest the exponent. This is
  the single most common real problem in practice.
- **Curvature** in residuals-vs-fitted: the model is missing a quadratic term,
  or the response genuinely bends. Confirm against the center-point curvature
  test.
- **Trend** in residuals-vs-run-order: something drifted. Ambient conditions,
  a reagent degrading, an instrument going out of calibration. If run order
  was recorded, this is detectable; if the runs were not randomised, it is
  indistinguishable from a factor effect, which is precisely why
  randomisation matters.
- **Outliers**: investigate before deleting. A point that is genuinely a
  transcription error can go. A point that is merely inconvenient stays. If
  one is dropped, say so in the report and show the analysis both ways.
- **Fat tails on the normal probability plot**: usually a transformation
  problem rather than a broken model.

A high R-squared with patterned residuals is worse than a low R-squared with
clean ones, because it is confidently wrong. Report R-squared adjusted rather
than raw R-squared, since raw R-squared rises whenever a term is added.
R-squared predicted, when available, is the honest one: a large gap between
adjusted and predicted means the model is fitting noise.

## Transformations

If `box_cox` suggests a transformation, apply it, refit and re-check the
residuals. Then remember to state results on the original scale, because
nobody thinks in log-yield. Predictions transformed back from a log fit are
medians rather than means, which is worth a footnote when the difference
matters.

## Optimisation

`optimize_responses` combines several responses via desirability, letting each
one be maximised, minimised or driven to a target with its own weight.

Three cautions worth passing to the user every time:

- The optimum is a **prediction from a model**, not an observation. Its
  uncertainty grows towards the edges of the design region, and an optimum
  sitting on a boundary usually means the true optimum is outside the region
  explored. That is a reason to run another stage, not to accept the boundary.
- **Never extrapolate** beyond the factor ranges studied. The model has no
  information there and will happily produce a confident number anyway.
- **Confirm.** Run the predicted optimum, at least in triplicate, and check it
  with `confirmation_test`. If the confirmation runs fall outside the
  prediction interval, the model is wrong somewhere and the result should not
  be used until that is understood.

## Analysing a design you did not generate

Verify it first (see `verification.md`). If it turns out to be resolution II,
stop: the aliasing means the data cannot separate the effects, and any
analysis will produce numbers that look fine and mean nothing.

If the design is resolution III or IV, pull the `alias_structure` and report
the specific ambiguities alongside the effects. "The apparent B effect is
aliased with the C:D interaction" is exactly the sort of caveat that keeps a
user from over-committing to a wrong conclusion.
