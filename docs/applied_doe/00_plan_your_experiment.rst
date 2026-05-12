Plan your own experiment
========================

Adapted from Kevin Dunn's *Planning / discussing an experimental design:
a template* (CC BY 4.0, 2019-2026).  This page is the pre-flight
checklist for any of the modules that follow.  Fill it in *before*
running the first experiment.

.. tip::

   Most failed studies are not failures of execution.  They are
   failures of planning.  Five minutes with this template costs less
   than one wasted run.

Objective
---------

State, in one sentence, what these experiments are for.

::

    My objective with these experiments is to ___.

Outcome variables
-----------------

There is almost always more than one outcome worth measuring.  Capture
every one you can; you never know which will turn out to be the real
constraint.

For each outcome record:

- name, description, units
- typical value, typical noise level
- whether it is a *direct measurement* or a *calculation* (calculated
  outcomes have their own noise budget worth understanding)

.. note::

   If you would have to repeat the experiment to know how repeatable a
   measurement is, plan a small repeatability study first.  See
   Module 3 for the role of replicates and center points.

Type of study
-------------

Mark where this study sits on the scale below.  The four headings drive
very different design choices:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Ruggedness
     - Screening
     - Characterization
     - Optimization
   * - Confirm robustness.
     - Eliminate factors.
     - Quantify which factors matter, by how much.
     - Find the operating point that best meets the objective.

Modules 1 to 4 cover screening and characterization.  Module 5 covers
fractional screening; Modules 6 to 8 cover optimization.

Practicalities
--------------

- Time required per run: ___.
- Cost per run: ___.
- Are experiments grouped into batches?  If yes, what determines a
  batch?  (See Module 6 on *blocking*.)

Factors
-------

Build a table like this, one row per factor:

.. list-table::
   :header-rows: 1
   :widths: 6 14 22 16 12 12 12 12 12

   * - Letter
     - Name
     - Description
     - Numeric or categorical?
     - Extreme min
     - Typical min
     - Average value
     - Typical max
     - Extreme max
   * - A
     -
     -
     -
     -
     -
     -
     -
     -
   * - B
     -
     -
     -
     -
     -
     -
     -
     -

For each factor, also record:

- Is it **hard to change** in sequential order?  (If yes, you may need
  to group runs and treat the factor as a block.)
- Is it forced to be the same across an entire batch?  (Same as
  *blocking* with an unavoidable structure.)
- What does your physical understanding say each factor should do to
  the outcome?  (Predicting outcomes *before* running runs is the
  single best habit in DoE - see Modules 6 and 7.)
- Is it in **limited supply** (a reagent you have only enough of for
  N runs)?  Critical input for blocking.

Disturbances
------------

List anything that can influence the system but that you cannot fully
control.  Examples:

- Operator experience (less experienced versus more experienced).
- Batch of raw materials.
- Ambient humidity, temperature, atmospheric pressure.
- Equipment that degrades over time (e.g. a sensor drifting between
  experiments).

For each disturbance, decide:

- Can you **measure** it?  If yes, record it as a **covariate** and
  include it in the analysis (Module 5).
- Can you neither measure nor control it?  Treat it as a true
  **disturbance** and design to minimize its bias (randomize run
  order, see Module 6).
- Is it a *nuisance factor* you can control?  Block on it.

What success looks like
-----------------------

Before the first run, write down:

- The **target value** of each outcome (or the direction: maximize /
  minimize).
- The **threshold** at which you would stop the study.
- The **budget** in runs.
- The **next decision** the data will inform.

If you cannot answer those four questions before running anything, the
study is not yet ready to start.

A worked example
----------------

The yogurt 2x2 study in Module 1 fills in this template as follows:

::

    Objective:   maximize tasting-panel mouth-feel score (1-10).
    Outcome:     mouth-feel score; coarse, integer-valued, noise approx 1 unit.
    Type:        characterization (find the recipe that maximizes taste).
    Practicalities: ~24 hrs per run (fermentation), no monetary cost,
                 no batching constraint.
    Factors:
        A: fat content of starter (numeric, 0% to 2%).
        B: fermentation time (numeric, 10 to 16 hrs).
    Disturbances: room temperature; bacterial-culture batch (block).
    Success:     a recipe scoring >= 9 reliably; next decision is
                 whether to put it into the production schedule.

Six lines.  That is the kind of artefact that survives the first
review meeting and the last.

See also
--------

- :doc:`01_two_factor_mindset` - first worked solution that uses the
  template above.
- :doc:`06_power_and_evaluation` - blocking, covariates, and the
  trade-off table.
- :doc:`08_multiresponse_confirmation` - the full course-wide concept
  review.
