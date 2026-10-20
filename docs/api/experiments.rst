Designed Experiments
====================

.. automodule:: process_improve.experiments.structures
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: process_improve.experiments.designs
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: process_improve.experiments.factor
   :members: Factor, Constraint, Response, ResponseGoal, DesignResult

.. automodule:: process_improve.experiments.designs_screening
   :members: dispatch_fractional_factorial, defining_relation, dispatch_plackett_burman, plackett_burman_runs, dispatch_taguchi

.. automodule:: process_improve.experiments.designs_response_surface
   :members: dispatch_ccd, orthogonal_alpha, blocking_alpha, ccd_center_points, dispatch_box_behnken, dispatch_dsd, dsd_conference_order, dsd_run_count

.. automodule:: process_improve.experiments.designs_optimal
   :members: dispatch_d_optimal, dispatch_i_optimal, dispatch_a_optimal

.. automodule:: process_improve.experiments.designs_mixture
   :members: dispatch_mixture

.. automodule:: process_improve.experiments.pseudocomponents
   :members: to_pseudocomponents, from_pseudocomponents

.. automodule:: process_improve.experiments.designs_constrained
   :members: constrained_optimal_design, ConstrainedOptions, Criterion, parse_constraint, rename_factors, build_candidates, model_matrix, fedorov_exchange

.. automodule:: process_improve.experiments.region
   :members: DesignRegion

.. automodule:: process_improve.experiments.separating
   :members: separating_constraint, constraint_through_cuts, NotSeparableError
   :show-inheritance:

.. automodule:: process_improve.experiments.designs_mixture_constrained
   :members: constrained_mixture_design, extreme_vertices, mixture_candidates, mixture_inequalities, scheffe_matrix

.. automodule:: process_improve.experiments.designs_supersaturated
   :members: dispatch_supersaturated, hadamard, e_s2, e_s2_lower_bound, n_fully_aliased

.. automodule:: process_improve.experiments.designs_space_filling
   :members: space_filling_design

.. automodule:: process_improve.experiments.designs_omars_ilp
   :members: generate_omars, solve_omars_ilp, OmarsSearchReport
   :show-inheritance:

.. automodule:: process_improve.experiments.designs_factorial
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: process_improve.experiments.crossed_arrays
   :members: cross_arrays

.. automodule:: process_improve.experiments.designs_three_level
   :members: dispatch_three_level_fraction

.. automodule:: process_improve.experiments.three_level_components
   :members: interaction_components

.. automodule:: process_improve.experiments.models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: process_improve.experiments.omars
   :members: analyze_omars, OmarsResult

.. automodule:: process_improve.experiments.marginal_means
   :members: marginal_means_prediction

.. automodule:: process_improve.experiments.least_squares_means
   :members: least_squares_means

.. automodule:: process_improve.experiments.multistratum
   :members: multistratum_anova, error_terms

.. automodule:: process_improve.experiments.visualization.api
   :members: visualize_doe, main_effects_plot

.. automodule:: process_improve.experiments.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: process_improve.experiments.compare
   :members:
   :show-inheritance:

.. automodule:: process_improve.experiments.true_process
   :members:
   :show-inheritance:

.. automodule:: process_improve.experiments.analysis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: process_improve.experiments.gaussian_process
   :members:
   :show-inheritance:

.. automodule:: process_improve.experiments.augment
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: process_improve.experiments.optimization
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: process_improve.experiments.trade_off
   :members:
   :show-inheritance:

.. automodule:: process_improve.experiments.omars_trade_off
   :members:
   :show-inheritance:

.. automodule:: process_improve.experiments.simulations
   :members:
   :show-inheritance:

.. automodule:: process_improve.experiments.datasets
   :members:
   :show-inheritance:

.. automodule:: process_improve.experiments.reference_catalogue
   :members:
   :show-inheritance:

.. automodule:: process_improve.experiments.design_catalog
   :members:
   :show-inheritance:

.. automodule:: process_improve.experiments.design_measures
   :members:
   :show-inheritance:

Run Sheets
~~~~~~~~~~

.. automodule:: process_improve.experiments.run_sheet
   :members: run_sheet, read_run_sheet

Strategy Recommender
~~~~~~~~~~~~~~~~~~~~

.. automodule:: process_improve.experiments.strategy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: process_improve.experiments.strategy.engine
   :members: recommend_strategy
   :show-inheritance:

The Next Round
~~~~~~~~~~~~~~

.. automodule:: process_improve.experiments.next_step
   :members: recommend_next_step, ACTIONS

Simulated Studies
~~~~~~~~~~~~~~~~~

.. automodule:: process_improve.experiments.simulate
   :members: simulate_design_performance

.. automodule:: process_improve.experiments.strategy.models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: process_improve.experiments.strategy.domain_templates
   :members: get_domain_template
   :show-inheritance:

.. automodule:: process_improve.experiments.strategy.budget
   :members:
   :undoc-members:
   :show-inheritance:
