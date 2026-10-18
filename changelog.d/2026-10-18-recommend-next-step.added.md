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
- **The browser app says what the results call for next.** Once every run of a study has its result,
  Learn's "The next step" opens with the step `recommend_next_step` takes for the response's goal,
  and one button that carries it out: the runs added to the run sheet as a block of their own,
  confirmation runs at the predicted optimum, or a new study with its ranges moved or widened. Why,
  and what to do with the runs, are folded beneath it. The study's blocks are fitted as rounds, and
  confirmation runs already made are judged, so the step follows the study round by round to its
  confirmed optimum.
