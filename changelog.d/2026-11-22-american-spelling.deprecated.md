- **Public names spelled the British way have American names** (#1232). Each former name still works until 3.0,
  with a `DeprecationWarning` that names its replacement:
  - `process_improve.batch.plotting.colours_per_batch_id`, and its `use_default_colour` and `colour_map`: use
    `colors_per_batch_id`, `use_default_color` and `color_map`.
  - `MultiTagPlotSettings.colour_map`, also as the `"colour_map"` key of `plot_multitags(settings=...)`: use
    `color_map`.
  - `get_rgba_from_triplet(incolour=...)` and `generate_one_frame(animation_colour_assignment=...)`: use `incolor`
    and `animation_color_assignment`.
  - `process_improve.chemistry.normalisation_check`: use `normalization_check`.
  - `process_improve.experiments.design_catalog.normalise_family`: use `normalize_family`.
  - `process_improve.experiments.reference_catalogue.reproduce_reference_catalogue`: use
    `reproduce_reference_catalog`.
  - `process_improve.multivariate.UncentredDataWarning`: use `UncenteredDataWarning`. The two names are the same
    class, so a warning filter or a `pytest.warns` written with either name matches both.
  - The `warn_on_uncentred` keyword of `PLS`, `PLSDA` and `PRM`: use `warn_on_uncentered`.
  - `process_improve.univariate.least_favourable_means`: use `least_favorable_means`.
