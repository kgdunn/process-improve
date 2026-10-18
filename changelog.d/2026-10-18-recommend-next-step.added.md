- **`recommend_next_step` decides the next round of a designed experiment from its results.** Pass
  every run so far (and any confirmation runs); it returns one action (stop, confirm, add centre
  points, replicate, widen the ranges, de-alias, augment to a response surface, add runs for the
  model, move along the path of steepest ascent, or investigate), the reasons with their numbers,
  the runs to make in the factors' own units, and how to judge them. It is also an MCP tool.
  Whether anything stands out from the noise is one test whose critical value is simulated for the
  design, so pure noise raises a false alarm in one study in twenty, and Lenth's method uses
  simulated critical values (with seven contrasts the individual one is 2.31 PSE, not the 3.76 of
  its t approximation). Followed round by round in simulation, from a screening design to `stop`,
  it reached the true optimum in every one of nine kinds of study.
