- **`defining_relation(design)`: the defining relation of a fraction on request, however many
  generators it has** (`process_improve.experiments`, #788 A9). A design lists its relation only up
  to ten generators (1023 words); past that, `metadata["defining_relation_note"]` now says so, and
  `defining_relation(design)` works out every word from the generators, in the design's own form
  (`"I=ABE"`, `"I=-ABD"`). The saturated 2^(15-11) of Montgomery's Table X gets its 2047 words.
