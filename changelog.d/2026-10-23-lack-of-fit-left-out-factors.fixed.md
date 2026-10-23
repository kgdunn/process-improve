- **Lack of fit and the curvature test take pure error from runs alike in every factor of the design, not only
  in the model's factors.** A factor left out of a formula used to make runs that differ only in it count as
  replicates, so its effect went into pure error, and a model fitted to every distinct setting of the factors it
  names reported no degrees of freedom for lack of fit. `analyze_experiment` now counts as `evaluate_design`
  does, and reproduces Table 8-3 of Anderson and Whitcomb's *DOE Simplified*: B and AB left out of the confetti
  model are 2 degrees of freedom of lack of fit (F = 3.75, p = 0.1527) on 3 of pure error. A column the formula
  leaves out with a different value in every run, such as a run number, still splits no replicates.
