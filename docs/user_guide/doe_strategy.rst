Experimental Strategy Recommendation
=====================================

Before running any experiments, the most important question is: *"How should
I plan my experimental program?"*  The ``recommend_strategy`` function answers
this by generating a complete multi-stage experimental plan - screening,
optimization, and confirmation - using deterministic decision rules from
Montgomery, NIST, and the Stat-Ease SCOR framework.

The recommender is fully deterministic: identical inputs always produce
identical outputs.  There is no randomness and no LLM - just ~50 codified
rules that encode best practices from the DOE literature.

When to Use This Tool
---------------------

- **Before your first experiment** - plan the entire workflow upfront so that
  budget and time are spent efficiently.
- **When you have many candidate factors** - the tool decides whether a
  screening stage is needed and which design to use.
- **When budget is limited** - it allocates runs across stages to maximize
  information per experiment.
- **When working in a specialized domain** - domain-specific templates
  (fermentation, cell culture, pharma, etc.) adjust design choices and
  center-point requirements automatically.

Concepts
--------

Multi-stage workflows
~~~~~~~~~~~~~~~~~~~~~

Most experimental programs follow a three-stage sequence:

1. **Screening** - Identify the vital few factors from many candidates.
   Typical designs: Definitive Screening Design (DSD), Plackett-Burman, or
   fractional factorial.
2. **Optimization** - Fit a response surface model for the significant
   factors.  Typical designs: face-centred Central Composite Design (CCD),
   Box-Behnken, or D-optimal.
3. **Confirmation** - Run replicates at the predicted optimum to verify
   that the model predictions hold.

Each stage has *transition rules* that tell you what to do next based on
the results.  For example, after screening:

- 0–1 significant factors found: broaden factor ranges or check the
  measurement system.
- 2–5 significant factors: proceed to optimization.
- 6+ significant factors: sub-group factors or run additional screening.
- Curvature detected at center points: augment the factorial to a CCD.

Six or more continuous factors are screened with a DSD by default. Its main
effects are orthogonal to every two-factor interaction and square, whereas in a
12-run Plackett-Burman design each interaction biases the main effects of the
factors outside it by a third of its size, which can hide a real factor.

Conditional stages
~~~~~~~~~~~~~~~~~~

An optimisation stage is often not needed. When the screening design can test
for curvature (a DSD, or a two-level design with centre runs), the optimisation
stage after it is *conditional*: ``stage["conditional"]`` is ``True`` and
``stage["run_if"]`` names the evidence that calls for it and the analysis that
gives that evidence. Without that evidence, the screening model already locates
the best settings and the study goes straight to confirmation.
``minimum_total_runs`` counts the runs when no conditional stage is called for,
and ``total_estimated_runs`` counts them all. A screening design that cannot
test for curvature, such as a Plackett-Burman design with a categorical factor,
is followed by an optimisation stage that is always run.

Budget allocation
~~~~~~~~~~~~~~~~~

When a budget is specified, runs are allocated across stages using the
25-40-55-15 framework (Montgomery / Stat-Ease):

- **Screening**: 25–40% of the budget
- **Optimization**: 40–55% of the budget
- **Confirmation**: 5–15% of the budget

Domain templates can shift these weights.  For example, fermentation
allocates more to optimization (50%) because biological variability demands
extra center points for reliable error estimation.

Quick Start
-----------

A 7-factor fermentation problem with a budget of 40 runs:

.. code-block:: python

   from process_improve.experiments.factor import Factor, Response
   from process_improve.experiments.strategy import recommend_strategy

   factors = [
       Factor(name="Temperature", low=25, high=40, units="degC"),
       Factor(name="pH", low=5.0, high=7.5),
       Factor(name="Glucose", low=10, high=50, units="g/L"),
       Factor(name="Yeast extract", low=1, high=10, units="g/L"),
       Factor(name="Agitation", low=100, high=400, units="rpm"),
       Factor(name="Aeration", low=0.5, high=2.0, units="vvm"),
       Factor(name="Inoculum", low=2, high=10, units="%v/v"),
   ]
   responses = [Response(name="Yield", goal="maximize", units="g/L")]

   strategy = recommend_strategy(
       factors=factors,
       responses=responses,
       budget=40,
       domain="fermentation",
   )

   for stage in strategy["stages"]:
       print(f"Stage {stage['stage_number']}: {stage['stage_name']}")
       print(f"  Design: {stage['design_type']}, Runs: {stage['estimated_runs']}")
       print(f"  Purpose: {stage['purpose']}")

This outputs::

   Stage 1: Screening
     Design: plackett_burman, Runs: 11
     Purpose: Screen 7 candidate factors to identify the vital few.
   Stage 2: Optimization
     Design: ccd, Runs: 19
     Purpose: Fit quadratic response surface model for the 3 significant factors. ...
   Stage 3: Confirmation
     Design: replicates_at_optimum, Runs: 3
     Purpose: Run replicates at the predicted optimum to verify the model predictions. ...

The engine selected Plackett-Burman screening (the fermentation domain
default) with 3 centre runs, a face-centred CCD for response surface
optimization, and 3 confirmation replicates - all within the 40-run budget.
The CCD is conditional on the centre runs showing curvature, so the study needs
14 runs when they do not and 33 when they do.

Interpreting the Output
-----------------------

``recommend_strategy`` returns a dictionary with these keys:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Key
     - Description
   * - ``stages``
     - Ordered list of experimental stages.  Each stage contains
       ``stage_number``, ``stage_name``, ``design_type``, ``design_params``,
       ``factors``, ``estimated_runs``, ``purpose``, ``success_criteria``,
       ``transition_rules``, ``conditional`` and ``run_if`` (see
       `Conditional stages`_).  ``design_type`` and ``design_params`` are a
       ``generate_design`` call: ``generate_design(factors,
       design_type=stage["design_type"], **stage["design_params"])`` builds the
       stage.  The confirmation stage (``"replicates_at_optimum"``) is the
       exception: its runs are replicates at the optimum that
       ``optimize_responses`` finds.
   * - ``total_estimated_runs``
     - Sum of estimated runs across all stages, the conditional ones included.
   * - ``minimum_total_runs``
     - Sum of estimated runs across the stages that are not conditional.
   * - ``budget_allocation``
     - Dictionary mapping stage names to allocated run counts.
   * - ``reasoning``
     - Step-by-step explanation of the decision logic.
   * - ``assumptions``
     - Key assumptions underlying the recommendation (e.g. factor ranges
       are wide enough, measurement system is adequate).
   * - ``risks``
     - Potential issues and warnings (e.g. tight budget, split-plot
       requirements).
   * - ``alternative_strategies``
     - Brief descriptions of other approaches worth considering.
   * - ``strategy_id``
     - Deterministic hash of the input - same inputs always produce the
       same ID.
   * - ``domain``
     - The application domain used.
   * - ``detail_level``
     - ``"novice"`` or ``"intermediate"``.

To inspect transition rules after screening:

.. code-block:: python

   for rule in strategy["stages"][0]["transition_rules"]:
       print(f"If {rule['condition']}:")
       print(f"  -> {rule['action']}")
       print(f"  Otherwise -> {rule['fallback']}")

Working with Budget Constraints
-------------------------------

The budget parameter controls how many total runs are available.  The engine
adjusts stage complexity accordingly:

.. code-block:: python

   for b in [60, 25, 20, None]:
       result = recommend_strategy(
           factors=factors, responses=responses, budget=b, domain="fermentation"
       )
       designs = [stage["design_type"] for stage in result["stages"]]
       print(f"Budget={str(b):>4s}: {result['total_estimated_runs']:>2d} runs, {designs}")

::

   Budget=  60: 33 runs, ['plackett_burman', 'ccd', 'replicates_at_optimum']
   Budget=  25: 20 runs, ['dsd', 'replicates_at_optimum']
   Budget=  20: 20 runs, ['dsd', 'replicates_at_optimum']
   Budget=None: 33 runs, ['plackett_burman', 'ccd', 'replicates_at_optimum']

With a tight budget, the engine changes the designs, not only their run
counts: it drops center points, replaces a stage's design with a smaller one
(a Plackett-Burman screen, a D-optimal quadratic design), and plans a single
Definitive Screening Design when separate screening and optimization stages
cannot fit. It plans that DSD too, for six or more factors, when the reduction
would take the centre runs out of the screening design: the optimisation stage
would then have to be run whatever the screening showed, and one DSD both
screens and tests for curvature. The budget must hold the conditional stages
too, since the screening may call for them. ``estimated_runs`` is always the size of the design the
stage's ``design_params`` build, and ``result["risks"]`` says what was
reduced, or that the budget is below the smallest plan. When ``budget=None``,
the ideal designs are used without constraint.

Using Prior Knowledge
---------------------

If you already know something about which factors matter, pass a free-text
description via the ``prior_knowledge`` parameter.  The engine parses
keywords to set a confidence level:

- **High confidence** (0.9): "confirmed", "validated", "published",
  "well-established"
- **Medium confidence** (0.7): "literature suggests", "preliminary data",
  "pilot study"
- **Low confidence** (0.4): "suspect", "expected", "based on theory"
- **No knowledge** (0.1): "no prior data", "first time", "exploratory"

High confidence (>= 0.8 with supporting data) skips the screening stage
entirely:

.. code-block:: python

   # No prior knowledge - full screening
   s1 = recommend_strategy(factors=factors, budget=40, domain="fermentation")
   print(f"No prior: {len(s1['stages'])} stages")

   # Low confidence - still screens
   s2 = recommend_strategy(
       factors=factors,
       budget=40,
       domain="fermentation",
       prior_knowledge="We suspect Temperature and pH are important.",
   )
   print(f"Low confidence: {len(s2['stages'])} stages")

   # High confidence - screening skipped
   s3 = recommend_strategy(
       factors=factors,
       budget=40,
       domain="fermentation",
       prior_knowledge=(
           "Published and validated results confirm Temperature " "and pH are significant."
       ),
   )
   print(f"High confidence: {len(s3['stages'])} stages")

::

   No prior: 2 stages
   Low confidence: 2 stages
   High confidence: 1 stages

(With no response to optimise, seven factors get no optimisation stage.)

Domain-Specific Strategies
--------------------------

The ``domain`` parameter selects a domain template that adjusts screening
design preferences, RSM design choices, center-point counts, and budget
weights.  Eight domains are available:

.. list-table::
   :widths: 22 30 48
   :header-rows: 1

   * - Domain
     - Screening / RSM preference
     - Notes
   * - ``"fermentation"``
     - Plackett-Burman / CCD
     - Extra center points (5+) for biological variability.
   * - ``"cell_culture"``
     - DSD / Box-Behnken
     - Minimizes runs for expensive, slow experiments (14–21 days).
   * - ``"pharma_formulation"``
     - DSD / Face-centered CCD
     - ICH QbD framework; design space definition for regulatory submissions.
   * - ``"food_science"``
     - Fractional factorial / BBD
     - Mixture handling; avoids extreme factor combinations.
   * - ``"extraction"``
     - Fractional factorial / CCD
     - Strongly quadratic responses; a rotatable CCD where the ranges are not hard limits.
   * - ``"analytical_method"``
     - Fractional factorial / CCD
     - AQbD / ICH Q2/Q14; includes robustness study stage.
   * - ``"bioprocess"``
     - Plackett-Burman / CCD
     - Scale-up considerations for bench-to-production transfer.
   * - ``"general"``
     - Rule-engine defaults
     - No domain-specific adjustments.

Comparing two domains on the same factors shows how design choices differ:

.. code-block:: python

   for domain in ["fermentation", "cell_culture"]:
       result = recommend_strategy(factors=factors, budget=40, domain=domain)
       screening = result["stages"][0]
       print(
           f"{domain:>15s}: {screening['design_type']}, "
           f"{screening['estimated_runs']} screening runs"
       )

::

   fermentation: plackett_burman, 11 screening runs
   cell_culture: dsd, 17 screening runs

Fermentation uses Plackett-Burman (efficient, many-factor screening, here with
3 centre runs), while
cell culture uses a Definitive Screening Design because it combines screening
and curvature detection in a single stage - saving an entire experimental
cycle when each run takes 2–3 weeks.

Hard-to-Change Factors
----------------------

When some factors are expensive or time-consuming to reset between runs
(e.g. reactor temperature, equipment configuration), flag them with
``hard_to_change_factors``.  ``generate_design`` builds split-plot structure in
the optimal designs, so each stage that contains a hard-to-change factor becomes
a D-optimal split-plot design of the same size, with ``hard_to_change`` naming
its whole-plot factors:

.. code-block:: python

   result = recommend_strategy(
       factors=factors,
       responses=responses,
       budget=40,
       domain="fermentation",
       hard_to_change_factors=["Temperature"],
   )

   for stage in result["stages"]:
       params = stage["design_params"]
       if "hard_to_change" in params:
           print(f"{stage['stage_name']}: {stage['design_type']}, {params['model_type']} model")
           print(f"  Whole-plot (hard to change): {params['hard_to_change']}")

::

   Screening: d_optimal, main_effects model
     Whole-plot (hard to change): ['Temperature']
   Optimization: d_optimal, quadratic model
     Whole-plot (hard to change): ['Temperature']

With split-plot designs, runs are grouped within whole-plot factor levels
to minimize the number of hard-to-change factor resets. The design records each
run's whole plot in a ``WholePlot`` column. The output risks include a reminder
that standard ANOVA gives incorrect p-values for split-plot experiments: runs in
one whole plot share its error, so ordinary least squares makes the
hard-to-change factors look more significant than they are. Analyse the results
with ``analysis_type="split_plot"``, a restricted maximum likelihood (REML) fit
that tests each factor against the right error; see :ref:`split-plot-analysis`.

Multiple Responses
------------------

When optimizing for more than one response, define each with its own goal:

.. code-block:: python

   responses = [
       Response(name="Yield", goal="maximize", units="g/L"),
       Response(name="Purity", goal="maximize", units="%"),
       Response(name="Cost", goal="minimize", units="USD/kg"),
   ]

   result = recommend_strategy(
       factors=factors,
       responses=responses,
       budget=40,
       domain="fermentation",
   )

The strategy structure is the same - the engine plans the experimental
stages needed to build models for all responses simultaneously.  After
running the experiments, fit each response with
:func:`~process_improve.experiments.analyze_experiment` and
``coding="coded"``, and pass the results to
:func:`~process_improve.experiments.optimize_responses` with desirability
functions to find the best trade-off across responses.

The optimiser evaluates the models at coded settings, from -1 to +1, so it
needs coefficients on that scale. Fitted in actual units (degrees, minutes,
bar), a model's coefficients would be searched over -1 to +1 of those units,
so the optimiser refuses them and says how to refit. Coded coefficients
carry the ranges they were coded from, so the optimum comes back in actual
units as well. When the data's extremes are not the levels that should be
-1 and +1, as with a central composite design's axial runs, give the ranges
instead: ``coding={"T": {"low": 150, "high": 200}, ...}``.

See Also
--------

- :doc:`/api/experiments` - Full API reference for all DOE functions.
- :func:`~process_improve.experiments.generate_design` - Generate the actual
  design matrix once you know which design to use.
- :func:`~process_improve.experiments.analyze_experiment` - Analyze the
  results after running experiments.
- :func:`~process_improve.experiments.optimize_responses` - Find optimal
  factor settings for single or multiple responses.
