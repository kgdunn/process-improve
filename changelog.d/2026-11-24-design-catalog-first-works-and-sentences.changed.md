- **Every family of designs in the catalog names the work that first published it (#1479).** The Latin square,
  the randomized blocks and the split plot had none: `catalog_families()` gives Fisher (1925), *Statistical
  Methods for Research Workers*, with the section of each, and a family's card on the page shows it.
- **The knowledge base's lists for its first seven design types are sentences.** `suitable_for`,
  `disadvantages` and the `method` of a `can_augment_to` step of the full factorial, the fractional factorial,
  and the Plackett-Burman, central composite, Box-Behnken, definitive screening and OMARS designs were labels,
  such as "Initial exploratory experiments". Each is a sentence that says when the design is used or what goes
  wrong, as for the other design types, and `description["novice"]` uses no symbol it does not explain. The
  catalog's page shows them on a family's card, and `doe_knowledge` returns them.
