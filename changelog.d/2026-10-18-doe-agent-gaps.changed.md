- **`recommend_strategy` screens six or more continuous factors with a definitive screening design.**
  In a 12-run Plackett-Burman design each interaction biases the main effects of the factors outside
  it by a third of its size; in a simulated 8-factor study that hid a real factor in 35% of the
  sessions that followed the old plan. The DSD gets the centre runs that leave `analyze_omars` four
  degrees of freedom for error (with two, it missed an active factor in 6% of simulated studies;
  with four, in none), and a Plackett-Burman screen, where a domain prefers one, gets three centre
  runs to test for curvature. A budget that would strip the screening
  design's curvature test gets one DSD and the confirmation runs instead. Every CCD the plan proposes
  is face-centred, so no run leaves the factor ranges.
- **`analyze_experiment` fits main effects when its default model cannot be estimated.** With no
  `model` given, a design that cannot estimate the interactions model and does not alias its
  interactions in whole chains (a Plackett-Burman or definitive screening design) now gets the
  main-effects model, with a warning and `model_summary["model_note"]`, instead of 37 coefficients of
  which 25 were arbitrary. `model_selection` still searches the interactions, and a regular fraction
  keeps its alias-chain analysis.
