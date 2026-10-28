- **`recommend_next_step` no longer confirms a model that misses the repeated runs while interactions of its active
  factors with the others are untested.** When the model the terms were judged in does not fit the repeats, that
  lack of fit now holds for the smaller final model too, whose own test has less power; and the second look also
  tests the interactions of an active factor with every other factor (weak heredity, Holm's method). Before, a
  10-factor study could confirm, after a definitive screening design and 4 de-alias runs, an optimum below two
  runs it had already made.
