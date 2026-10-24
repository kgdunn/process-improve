- **`compare_designs` says how to compare over a region whose limits name a categorical level (#783).** It
  refused such a region by saying the designs lack the categorical factor, although designs with one are refused
  too, since the comparison is over numeric factors. It now says the limits give one region per allowed
  combination of levels, and to compare over each, such as `region.given(catalyst='A')`. `augment_design`'s
  refusal of the same region points to `region.given(...)` as well, and `separating_constraint`'s docstring says
  how to write a proposed limit with a level.
