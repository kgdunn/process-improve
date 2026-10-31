- **`augment_design(..., "foldover", generators=...)` and `"semifold"` keep the signs of the defining
  words.** A design with a negated generator (`D=-AB`) has negative words in its defining relation,
  but the fold reported every word that survived as positive: `I=ABCG` where every run of the
  combined design has `ABCG = -1`. An alias chain read from it then had the wrong sign, so a
  difference of effects was read as a sum. `defining_relation` and the explanation now write a
  negative word as `I=-ABCG`, as `evaluate_design` already does. Designs built from positive
  generators only are unchanged. (#966)
