# Verifying a design

Read this before handing any design to a user, and before analysing any design
you did not generate.

## Why this step is not optional

A design matrix is a grid of plus and minus signs. Nothing about looking at it
tells you whether the columns are balanced, whether two of them are secretly
the same, or whether a main effect is aliased with an interaction. Those
properties are what determine whether the experiment can answer its question,
and they are invisible to inspection.

Vazquez, Rother and Charles-Gonzalez (2026, arXiv:2512.17113) assessed how well
this goes when a language model writes the matrix instead of a catalogue. They
had GPT and Gemini models construct two-level fractional factorial designs with
8, 16 and 32 runs and 4 to 26 factors, and compared the results against the
best-known designs on resolution and minimum aberration. The models construct
optimal designs reliably up to about eight factors, and degrade beyond that.
A design matrix that degrades this way does not look wrong on the page.

So verify. It costs one command.

## How to verify

```bash
python scripts/verify_design.py design.csv
python scripts/verify_design.py design.csv --require-resolution 4 --expect-runs 16
```

Or through the registry, which also gives you the other quality metrics:

```bash
python scripts/doe_tool.py call evaluate_design --input '{
  "design_matrix": [...],
  "metric": ["moment_aberration", "resolution", "alias_structure", "d_efficiency", "vif"]
}'
```

## Resolution, and how to explain it

Report the number *and* what it costs. The number alone means nothing to most
users.

| Resolution | What to say |
|---|---|
| I | Not level-balanced: some column has unequal numbers of low and high settings. This is not a usable factorial design. Regenerate it. |
| II | A main effect is aliased with another main effect. Two columns carry identical information and their effects can never be separated. Do not run this design. |
| III | Main effects are separable from each other but confounded with two-factor interactions. Usable for a first screening pass, as long as the user understands a large apparent main effect may really be an interaction. |
| IV | Main effects are clear of each other and of two-factor interactions; some pairs of two-factor interactions are aliased with each other. The standard screening choice. |
| V | Main effects and all two-factor interactions are clear of each other. Good enough to model with, not just to screen. |
| VI+ | Effectively no aliasing among anything the user cares about. |

Resolution I or II means stop. Do not analyse data from such a design as if it
were valid, and do not let it reach a lab.

## Minimum moment aberration

Two designs can share a resolution and still differ in quality: resolution
counts only the *shortest* word, so it says nothing about how many words of
that length there are. Minimum aberration breaks the tie by counting them.

Classical minimum aberration reads the word-length pattern off the defining
relation, which means it needs the design's generators. That rules out exactly
the case that matters here, a matrix someone handed you.

Minimum *moment* aberration (Xu, 2003) works from the matrix alone. It ranks
designs by the moments of the distribution of pairwise similarities between
runs, is equivalent to minimum aberration for regular designs, extends to
non-regular ones, and yields the design's strength and hence its resolution
for any two-level matrix. `moment_aberration` in this library implements it.

To choose among candidates of the same size, take the one whose pattern is
smallest read left to right:

```bash
python scripts/verify_design.py candidate_a.csv candidate_b.csv --compare
```

The report also names the first moment that falls short of its lower bound.
That order is where the design's aliasing begins, and it is one more than the
strength.

## Other metrics worth reporting

Pull these from `evaluate_design` when they are relevant:

- **`alias_structure`** - the actual alias chains, e.g. `A = A + BCD + ...`.
  Worth showing whenever the resolution is III or IV, because it tells the
  user which specific interpretations are ambiguous rather than leaving it
  abstract.
- **`power`** - the probability of detecting an effect of a given size. If the
  user can state the smallest effect worth finding and a noise estimate, this
  is the metric that says whether the run budget is adequate. A design with
  40% power will probably waste the whole budget. Report it *before* the runs
  happen, when it can still change the plan.
- **`vif`** and **`condition_number`** - for optimal and non-orthogonal
  designs. VIFs of 1 mean perfectly orthogonal. Above about 5 the coefficient
  estimates start blurring into each other.
- **`d_efficiency`** / **`i_efficiency`** - for comparing optimal designs.
  D for estimating coefficients precisely, I for predicting well across the
  region. Optimisation usually wants I.
- **`degrees_of_freedom`** - a quick check that the design can actually fit
  the intended model. Zero residual degrees of freedom means no F-test, and
  the analysis will need Lenth's method.

## Verifying a design that came from elsewhere

When the user pastes a design in:

1. Get it into a CSV with one column per factor, one row per run, and no
   response column (or select factors with `--factors A,B,C`).
2. Run `verify_design.py` on it.
3. Report the resolution the *matrix* has, and if the user told you what it
   was supposed to be, say plainly whether those agree.
4. If it fails, do not patch it by hand. Generate a correct one and show the
   difference.

Sanity checks that catch most bad matrices immediately, and are worth
mentioning when they fire:

- Every column should have equal numbers of low and high settings.
- No two columns should be identical, nor exact negatives of each other.
- Run count for a regular fractional factorial is a power of two.
- No duplicated rows unless replication was intended.
- No missing cells.

## Center points and mixed-level designs

`moment_aberration` is defined for two-level designs and will decline anything
else with a clear message. That is not a limitation to work around: verify the
two-level core of the design, then add center points afterwards. For
genuinely mixed-level or response-surface designs, use `d_efficiency`,
`i_efficiency`, `vif` and `power` instead, since resolution is not defined for
them.
