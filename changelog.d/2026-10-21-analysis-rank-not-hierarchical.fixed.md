- **`analyze_experiment` ranks a model that is not hierarchical as it is given.** The design's rank was
  taken with every factor coded to -1/+1, which re-expresses a hierarchical model but changes one that is
  not: a factor nested in another through the other's 0/1 indicators, `L + I1:o + I2:o`, lost a column once
  coded (`I2` becomes the negative of `I1`), so a full-rank model was reported rank-deficient and its terms
  as aliased. The rank is now the model's own, its columns as given scaled to unit length, whenever the
  coded columns do not span them; a hierarchical model is ranked as before.
