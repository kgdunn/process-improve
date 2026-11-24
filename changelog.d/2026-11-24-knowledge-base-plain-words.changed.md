- **The knowledge base's lists for a design type are in plain words.** `suitable_for`, `disadvantages` and the
  `method` of a `can_augment_to` step no longer use a symbol or an abbreviation that they do not explain: "More
  runs than BBD for same k" is "More runs than a Box-Behnken design for the same number of factors". The design
  catalog's page (factori.al/catalog) shows these lists on a family's card, to a reader who may meet the design
  for the first time (#1475).
- **A full factorial "cannot detect curvature without center points".** The knowledge base said it "cannot
  estimate quadratic effects without center points"; center points show that curvature is present, and do not
  estimate a quadratic effect.
- **The catalog's cards for Taguchi arrays, mixtures and supersaturated designs take their words from the
  knowledge base**, as the cards of the factorial and response surface families do, and say which families the
  design can lead to. `catalog_families()` finds a family's entry by the `generate_design` type that makes the
  family, so a family and its design type need not share a name.
