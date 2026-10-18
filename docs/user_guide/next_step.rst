The Next Round of an Experiment
===============================

A study is a sequence of rounds, and after every round the question is the same: given every
run so far, what should the next round be? ``recommend_next_step`` answers it with one action,
the reasons with their numbers, the runs to make in the factors' own units, and what to do with
them once they are made. Called after every round, it carries a study from its first screening
design to a confirmed optimum.

It plans nothing in advance: ``recommend_strategy`` plans the stages before the first run, and
``recommend_next_step`` decides each next one from the results. The decision is deterministic:
the same runs always give the same answer.

A worked example
----------------

Eight factors are screened with a definitive screening design, and the process (unknown to the
experimenter) has three active main effects and one interaction:

.. code-block:: python

    from process_improve.experiments import Factor, TrueProcess, generate_design, recommend_next_step

    names = list("ABCDEFGH")
    factors = [Factor(name=n, low=-1, high=1) for n in names]
    runs = generate_design(factors, design_type="dsd", n_center_points=5).design_actual[names]
    truth = TrueProcess({"Intercept": 100, "A": 10, "C": 7, "G": -6, "A:C": 8}, sigma=3)
    runs["y"] = truth.simulate(runs, random_state=1).to_numpy()

    step = recommend_next_step(factors, runs, "y", ranges_can_move=False)
    print(step["action"], step["evidence"]["active_terms"])
    # confirm ['A', 'C', 'G', 'A:C']

The staged analysis of the design finds exactly the active terms, the best settings sit at
three edges of the ranges, and the ranges cannot move, so the next round is three confirmation
runs there. ``step["runs"]`` holds them, ``step["after"]`` says how to judge them, and
``step["evidence"]["optimum"]`` gives the prediction with the interval their mean should fall
in. Made and passed back as ``confirmation_runs``, runs whose mean falls inside it stop the
study.

With ``ranges_can_move=True`` the same runs give ``move_region`` instead: the slope at the
edges is many times the noise, so four runs along the path of steepest ascent come first.

The loop
--------

After every round, append the new runs to ``runs`` and call again, passing ``history``, the
actions so far. Some actions ask for more:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Action
     - What to do with its runs
   * - ``confirm``
     - Make them, and pass them as ``confirmation_runs``, not in ``runs``. After a miss the
       reply may ask for the confirmation again: add the runs that missed to ``runs``, and
       pass only the new ones as ``confirmation_runs``.
   * - ``add_center_points``, ``replicate``, ``de_alias``, ``augment_to_rsm``,
       ``add_runs_for_model``
     - Make them and append them to ``runs``.
   * - ``widen_ranges``
     - Set each factor's range to ``new_ranges`` and start the runs afresh with the ones given.
   * - ``move_region``
     - Make the runs in order and stop at the first that does worse than the one before. Centre
       new ranges, as wide as these, on the best run, and start a new design there.
   * - ``stop``, ``investigate``
     - The study ends, or waits for a look at the process.

Runs made in rounds
~~~~~~~~~~~~~~~~~~~

A process can drift between rounds: a new lot of raw material, a recalibrated instrument. Pass
the column that holds each run's round as ``block``, and every model fits a shift for each
round, so the drift neither biases the effects nor swells the noise. In a simulated study whose
second round ran 8 units (11 noise standard deviations) high, the blocks recovered the true
optimum; without them the noise estimate grew more than fourfold and the effects of A and B were
lost. The next runs are predicted as the process was in the latest round, since that is when
they are made.

The rules, in order
-------------------

1. **Confirmation runs** are judged by their mean against the prediction interval for a mean of
   that many runs. Inside it, the study stops, unless the goal still improves past an edge of a
   range that may move. A miss first gets the runs that show what the design could not (axial
   runs, a fold-over); else, when the model chosen again with the missed runs among its data
   has its optimum elsewhere, that optimum is confirmed; else a miss that may be chance is
   confirmed again and judged on the new runs alone, since pooling them with the runs that
   missed would carry the first miss into the second test. A gross miss (more than three
   half-widths of the interval), or a second one, is ``investigate``.
2. A model with fewer than two **degrees of freedom for error** cannot be judged: centre runs
   for a two-level design, else the runs made again.
3. **Nothing stands out** from the noise. With fewer than four degrees of freedom for error, a
   real effect can fail to stand out for want of them, so runs are added first. Otherwise the
   ranges are widened by half again (or the runs made again, when the ranges cannot move); if
   that was done before, the study stops: no factor moves the response by more than the noise.
4. An active term is **aliased with a plausible interaction**, one whose two factors are both
   active (strong heredity), so the runs cannot say which is real: fold the design over, or add
   runs chosen for the model that holds both. An interaction counts as aliased when the model's
   columns reproduce its column with an R squared of 0.9 or more: wholly in a regular fraction,
   and not in a Plackett-Burman or definitive screening design, whose partial aliasing the
   analysis separates.
5. **Curvature** the runs cannot model: axial runs, which make a face-centred central composite
   design and keep every run inside the ranges, or runs chosen for the full quadratic model.
   The curvature centre runs show could be any factor's square, and near an optimum a factor's
   linear effect vanishes while its square does not, so the axial runs go in every numeric
   factor, up to six, rather than in the active ones only.
6. The best settings sit at an **edge**, and the goal improves past it by more than twice the
   noise per coded unit: when the ranges may move, four runs along the path of steepest ascent.
7. Otherwise, **confirm** at the predicted optimum.

How the terms are judged
------------------------

Testing the terms a stepwise search picked, on the data that picked them, makes them look
stronger than they are. Each term is therefore tested in the largest model the runs can test it
in:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Runs
     - Judged by
   * - Enough runs for the full quadratic, the interactions or the main-effects model, with two
       degrees of freedom left for error
     - t-tests in the largest of these, with one column per chain of exactly aliased terms
   * - A saturated two-level design (a fraction, or a full factorial without repeats)
     - Lenth's method on all its contrasts, with simulated critical values
   * - A definitive screening design
     - its staged analysis (``analyze_omars``), every test at ``alpha``, with strong heredity
       for interactions
   * - A Plackett-Burman design
     - t-tests of every main effect and of the interactions a stepwise search (AICc) proposes,
       since main effects alone would leave an active interaction in the error, where it hides
       them

Interactions of the active factors that this could not test are then tested in the model of the
active terms. Lenth's own margins, a t quantile on a third as many degrees of freedom as
contrasts, are approximate: with seven contrasts the individual one is 3.76 pseudo standard
errors where the exact value is 2.31 (Ye and Hamada, 2000), and the simultaneous one is so wide
that two clear effects in an eight-run design rarely pass it. ``recommend_next_step`` uses
critical values simulated for the number of contrasts.

Whether anything stands out at all is one test at ``alpha``. The smallest p-value times the
number of terms (Bonferroni) sees one large effect; the F-test of the whole model sees several
moderate ones, which Bonferroni misses with few degrees of freedom for error. The statistic is
the smaller of the two, and its critical value is simulated for the design, so pure noise raises
a false alarm in one study in twenty.

How well it works
-----------------

Each study below was simulated 40 times on a known process with normal noise, following every
recommendation from the first design until the study stopped. The shortfall is how far the true
response at the final settings falls below the true optimum (inside the ranges when they cannot
move), in noise standard deviations.

.. list-table::
   :header-rows: 1
   :widths: 34 34 10 11 11

   * - Study
     - Path taken most often
     - Runs
     - Shortfall
     - Worst
   * - Adhesion (8 factors, three active, one interaction), DSD
     - confirm, stop
     - 25.0
     - 0.00
     - 0.06
   * - Adhesion, Plackett-Burman with three centre runs
     - confirm, stop
     - 19.2
     - 0.00
     - 0.00
   * - Adhesion, Plackett-Burman without centre runs
     - confirm, stop
     - 18.4
     - 0.03
     - 1.1
   * - Interior maximum, 2\ :sup:`3` with three centre runs
     - augment_to_rsm, confirm, stop
     - 23.2
     - 0.01
     - 0.04
   * - Interior minimum, 2\ :sup:`4-1` with three centre runs
     - augment_to_rsm, confirm, stop
     - 24.4
     - 0.17
     - 0.68
   * - Interior maximum, six-factor DSD
     - confirm, stop
     - 21.1
     - 0.15
     - 1.2
   * - Resolution III fraction, an active effect aliased with A:B
     - de_alias, confirm, stop
     - 19.6
     - 0.00
     - 0.00
   * - Optimum outside the ranges, 2\ :sup:`3` with three centre runs
     - augment_to_rsm, move_region, augment_to_rsm, confirm, stop
     - 47.4
     - 0.04
     - 0.37
   * - Pure noise, 2\ :sup:`4` with three centre runs
     - widen_ranges, stop
     - 38.2
     - 0.00
     - 0.00

Every one of the 360 studies stopped. The largest shortfalls come from a square missed at the
0.05 level in a design with few degrees of freedom for error: the confirmation runs then agree
with a model whose optimum sits at an edge rather than just inside it.
