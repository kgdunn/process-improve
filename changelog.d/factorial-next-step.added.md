- **Factorial's Learn says what to do next.** A new section, The next step, follows the analysis.
  When the best settings sit at an end of the ranges and the model predicts better past it, a table
  gives the ranges a next study would have, moved along the steepest ascent (or descent), and a
  button starts that study in Design with this study's brief and design. Learn also offers the ways
  of adding runs that suit the design, each with what it is for, through `augment_design`: a
  fold-over for a two-level design whose main effects are aliased with two-factor interactions,
  axial runs that make a two-level design fitting the interactions a face-centred central composite
  design, center runs for a two-level design with none, runs chosen for the model the study aims to
  learn (D-optimal), and every run made again when the model leaves fewer than three runs for the
  noise. A way whose runs would break a limit on the settings is not offered, and Learn says so.
  The runs added go at the end of the run sheet as a block of their own (center runs excepted,
  since their shift from the others is the curvature they test); the runs before keep their places
  and results, and a practice study's runs added are simulated from the same process and seed. The
  grown study is a design of runs as given, analysed with its blocks, and runs as given now keep
  their blocks when made again (`api_next_runs`, `api_add_runs`).
