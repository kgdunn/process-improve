- **The design catalog files 18 records in the family whose card describes them (#1478).** A word that was a
  rule by itself filed what only shared the word. Three crossed and nested classifications of Davies (1956)
  were orthogonal arrays, by the word "crossed": they are comparisons. Regular simplex designs were mixtures,
  by the word "simplex": four have no family, and the simplex-sum design is a central composite design. A
  fitted equation printed without its runs was in two families: all 23 are under "Comparison or regression".
  `catalog_records()` and the page (factori.al/catalog) give the new families.
- **Montgomery's supersaturated design for ten factors is its six runs.** The record
  `montgomery-supersaturated-12-run` held the table the design is taken from, 12 runs and 11 factors. That
  table is a record of its own, `montgomery-plackett-burman-12-run-sorted`, and the design is the same runs as
  `generate_design(..., design_type="supersaturated")` makes for ten factors.
- **Three statements of the knowledge base about response surface designs were not true of the designs the
  library builds.** A central composite design built on a full factorial has as many runs as a Box-Behnken
  design or more, not more for every number of factors: both have 27 for four. A Box-Behnken design does
  not lead to a central composite design: it has no run at a corner of the cube, so `can_augment_to` of
  `box_behnken` is empty.
  The `min_runs` of a mixture design gives one number: its formula is the model's terms and the centroid.
