# Choosing a design

Read this when deciding what to generate. The short version: the number of
factors and how much you already know decide the design, not the other way
round.

## The screening / optimisation split

This is the first fork, and getting it wrong is expensive.

**Screening** answers "which of these factors matter at all?" It is the right
first stage when there are more than about five candidate factors, or when the
user cannot confidently name the two or three that dominate. Screening designs
are cheap per factor, deliberately assume no curvature, and are content to
alias interactions. Their output is a shortlist, not a model.

**Optimisation** answers "what settings are best?" It needs curvature, so it
needs three or more levels per factor, which makes it expensive per factor. It
is only worth running on factors that screening has already shown to matter.

The single most common mistake is trying to do both at once with eleven
factors and a central composite design. That costs hundreds of runs to
estimate quadratic terms for factors that turn out to be inert. Screen first.

A useful budget split: about a third on screening, about half on optimisation,
the rest held back for confirmation and for the surprise that always turns up.

## Two-level designs (screening and factor effects)

| Design | Runs | Factors | When |
|---|---|---|---|
| Full factorial `2^k` | 2^k | 2-5 | Few factors, and you want every interaction cleanly |
| Fractional factorial `2^(k-p)` | 8, 16, 32 | 3-15 | The workhorse. Pick the resolution deliberately |
| Plackett-Burman | 12, 20, 24 | up to n-1 | Many factors, main effects only, run count not a power of 2 |
| Definitive screening (DSD) | 2k+1 or more | 4-12 | Screening *and* curvature detection in one design |

Beyond five factors a full factorial is almost never the right answer: 2^7 is
128 runs to estimate 7 main effects and a pile of high-order interactions
nobody believes in.

**Definitive screening designs** deserve more use than they get. They give
main effects clear of two-factor interactions, detect curvature, and cost
about 2k+1 runs. When a user has 6-10 continuous factors and wants one design
rather than two stages, a DSD is usually the best answer available.

## Resolution: what you are buying

Resolution is the length of the shortest word in the defining relation.
Practically:

- **III**: main effects clear of each other, aliased with two-factor
  interactions. Cheapest. Use only for a first pass with many factors, and
  only if the user accepts that a big apparent main effect might be an
  interaction.
- **IV**: main effects clear of each other *and* of two-factor interactions;
  pairs of two-factor interactions aliased with each other. The default
  screening choice. A resolution-IV design can be de-aliased later by folding
  over, which is why it is a safe place to start.
- **V**: main effects and all two-factor interactions clear of each other.
  Enough to build a model with, not just to screen.

Ask for resolution IV unless the run budget forbids it. The extra runs over a
resolution-III design are usually cheaper than the ambiguity they remove.

## Response surface designs (optimisation)

| Design | Runs (k factors) | When |
|---|---|---|
| Central composite (CCD) | 2^k + 2k + centers | The standard. Factorial core plus axial points |
| Box-Behnken | 3-level, no corners | When the corner combinations are impossible or unsafe |
| Optimal (D/I/A) | whatever you specify | Constrained regions, odd budgets, categorical factors |

**CCD** is the default when the factor region is a cube and the corners are
reachable. Its axial points sit outside the original ranges (unless
`alpha="face_centered"`), so check the extended settings are physically
possible before handing over the run sheet.

**Box-Behnken** never visits a corner, so it is the choice when "all factors
at maximum simultaneously" would break something.

**Optimal designs** are the fallback whenever the region is irregular: mixture
constraints, a forbidden corner, a fixed odd budget, or a mix of continuous
and categorical factors. Use `i_optimal` when the goal is prediction across
the region (the usual case for optimisation) and `d_optimal` when the goal is
estimating coefficients precisely.

## Mixture designs

Use `mixture` when the factors are proportions of a formulation that must sum
to a constant. Ordinary factorial designs are invalid here: the components are
not independent, because raising one necessarily lowers another.

## Split-plot: hard-to-change factors

If a factor is expensive or slow to reset (oven temperature, a reactor
changeover), full randomisation may be impractical. Pass
`hard_to_change_factors` to `recommend_strategy` and it will propose a
split-plot. The critical point to tell the user: a split-plot has *two* error
terms, and analysing it as if it were fully randomised will overstate the
significance of the hard-to-change factor. Record which factor it was.

## Center points

Include them, normally 3 to 5, whenever the factors are continuous. They cost
little and give:

- a pure-error estimate independent of the model, which makes lack-of-fit
  testable;
- a curvature check, telling you whether a two-level design is sufficient or
  whether the response is bending and needs a response surface;
- a drift check, if you spread them across the run order rather than clumping
  them at the start.

Note that a design with center points is no longer two-level, so
`moment_aberration` will decline it. Verify the two-level core, then add the
center points.

## Replication

Replication and center points are not the same thing. Replicates estimate
error at the design points; center points estimate it at the middle. An
unreplicated factorial has zero degrees of freedom for error, which forces
`lenth_method` at analysis time instead of an F-test. That is fine and normal
for screening, but if the user wants confidence intervals on effects they need
either replication or center points.

## Augmenting rather than restarting

When a first design comes back ambiguous, `augment_design` is usually cheaper
than a new design:

- **Fold-over** resolves the aliasing in a resolution-III design by adding a
  mirrored block, promoting it to resolution IV.
- **Axial points** upgrade a two-level factorial to a central composite, so a
  screening design becomes an optimisation design without discarding the runs
  already done.
- **Extra replicates** buy degrees of freedom for error when an effect sits
  frustratingly near the significance boundary.

This is the payoff for staging the budget: the second stage builds on the
first instead of replacing it.
