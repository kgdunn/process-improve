- **Factorial makes space-filling designs.** The design list offers Space-filling (Latin hypercube),
  a maximin Latin hypercube from `generate_design(design_type="maximin_lhs")`: runs spread evenly
  through the ranges for no model in particular, each factor at as many different settings as there
  are runs. It opens on ten runs a factor, the usual start for a model fitted to a computer
  experiment, and takes the reader's own number, replicates and blocks. Each setting is rounded to a
  tenth of the step between a factor's settings, so it can be set and stays in its step, and the
  Python code rounds it the same way. Learn analyses the runs with the model the study aims to learn,
  and the comparison of designs for a model leaves the design out. It takes numeric factors and no
  limits on the settings.
