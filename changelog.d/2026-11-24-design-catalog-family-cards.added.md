- **The design catalog (factori.al/catalog) says what a family of designs is.** A list that holds one family, as
  after a press on "Orthogonal array (Taguchi)" in the filters, has a card over it: what a design of the family is,
  when it is used, what to watch for, the families it can lead to, the work that first published it, and a link to
  it on the concept map. A family's name in an open row of the table is a button: it lists the family's designs of
  that row under the card, and "Show every ... design" on the card lists the family's designs of every size.
- **Nine families of the catalog have a description: orthogonal arrays, mixtures, optimal designs, supersaturated
  designs, Latin squares, randomized blocks, split plots, comparisons and "Other".** `catalog_families()` returned
  `""` for each, and the catalog named them and said no more. The words are fields of `Family` (`about`, `use_when`,
  `watch_for`) for the families the knowledge base of design types has no entry for.
- **`catalog_families()` returns `use_when` and `watch_for` for each family**: when a design of the family is used,
  and what to watch for, as lists of short phrases. For a family the knowledge base describes they are its
  `suitable_for` and `disadvantages`.
