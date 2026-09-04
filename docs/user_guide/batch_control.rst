Mid-course correction of batch processes
========================================

Replaying a golden batch's schedule is open-loop control, and the
:doc:`batch simulator <batch_simulator>` page measures what that costs.
This page closes the loop: predict the final quality of a *running* batch
from its initial conditions and its trajectories so far, and, when the
prediction falls short, re-optimise the remaining setpoint schedule. The
machinery lives in :mod:`process_improve.batch.control` and needs the
``control`` extra (``pip install 'process-improve[control]'``, which brings
in the `osqp <https://osqp.org>`_ QP solver).

Every number on this page is *executed*: the corrected schedule is fed back
into the simulator with the identical disturbance seed, so each gain is a
true same-batch counterfactual, not a model's opinion of itself. That
discipline matters. Simulation studies in this literature do execute their
corrections (Flores-Cerrillo and MacGregor rerun their non-linear nylon model
with the computed trajectories), but that step is unavailable on an operating
plant, so industrial results are reported as model predictions.

The pieces
----------

- :meth:`BatchPLS <process_improve.batch.BatchPLS>` relates the unfolded
  ``[Z | X]`` batch history to final quality, and
  :meth:`~process_improve.multivariate.PLS.project` estimates the scores of
  a batch whose future columns are still missing (trimmed score regression
  by default).
- :func:`~process_improve.batch.control.midcourse_correction` solves the
  correction as a convex program: track the target (or maximise quality),
  move as little as possible from the nominal remaining schedule, stay where
  the model has data (SPE and Hotelling's T2), and respect setpoint bounds
  and rate-of-change limits, with an optional knot parameterisation so the
  corrected schedule stays smooth. With the validity terms as penalties this
  is a quadratic program; the hard SPE and T2 caps are quadratic constraints,
  handled by an outer iteration on their penalty weights.
- :class:`~process_improve.batch.control.MidCourseCorrector` wraps the
  decision-point workflow. Its ``predict`` method answers the monitoring
  question at a decision point: the predicted final quality with a
  prediction interval built from the model's error *at that decision point*
  (the training batches re-projected under the same missingness pattern,
  against their measured quality), the SPE of the batch so far against its
  limit, and the condition number of the score estimator. Its ``correct``
  method adds the decision: the SPE validity gate (an out-of-family batch is
  not corrected, following Flores-Cerrillo and MacGregor, 2004), the
  no-correction dead band (correct only when the projected shortfall is
  significant against that interval, the practical lesson of Yabuki and
  MacGregor, 1997; the default of 1.0 asks that the whole interval fall
  short of the target), and per-decision-point limits built the same way
  (Garcia-Munoz, Kourti and MacGregor, 2004).
- :func:`~process_improve.batch.control.evaluate_control_policies` runs the
  whole comparison end to end.

Two findings from building this page shape the defaults. The historical
campaign must contain deliberate setpoint moves in the *shapes* the
controller will use: the simulator's ``historical`` policy therefore varies
its schedules with independent knot offsets, and corrections use the same
knot basis. And a single global linear model averages away a gain direction
that depends on the feed class (warmer mid-batch rescues a slow batch and
hurts a fast one), so the evaluation fits one model per feed class and
assigns a fresh batch to the nearest class centroid in standardised Z; with
the class ranges overlapping along the feed-quality axis that assignment is
right about 80% of the time, and a miss hands the batch the neighbouring
range's model.

The executed comparison
-----------------------

.. code-block:: python

    from process_improve.batch.control import evaluate_control_policies
    from process_improve.simulation import BioreactorSimulator

    result = evaluate_control_policies(
        BioreactorSimulator(), y_target=8.0, random_state=0
    )
    print(result.summary.round(3))

Measured on the default configuration (200 historical training batches, 40
fresh test batches, one decision point at sample 8, which is day 4 of 10;
seed 0; about seven minutes, dominated by the two ceiling policies):

===============  ======  ======  ======  ======
Policy           Mean    Sd      Min     Max
===============  ======  ======  ======  ======
replay           7.507   1.198   3.655   8.925
midcourse        7.709   0.781   5.717   8.925
oracle-from-k    7.828   0.631   6.108   8.925
adapted          7.824   1.013   4.573   9.309
===============  ======  ======  ======  ======

Four of the forty batches were corrected, all in the poorest feed class;
thirty-five were left alone by the dead band and one was stopped by the SPE
validity gate. Reading the table:

- The corrected batches gained between +1.53 and +2.67 g/L, mean +2.02 g/L,
  and none was harmed. The worst batch in the campaign rose from 3.66 to
  5.79 g/L.
- The campaign standard deviation fell from 1.20 to 0.78 g/L, a 35%
  reduction, with the mean up 0.20 g/L: mid-course correction works on the
  low tail, which is where the money is when the target is a floor.
- The **oracle-from-k** row re-optimises the remaining schedule of those
  same corrected batches against the simulator itself (the true process) at
  the same decision point: the ceiling for any mid-course scheme there. The
  data-driven correction captured 63% of the oracle's mean improvement; the
  rest is the price of an empirical model confined to the region its
  history explored.
- The **adapted** row runs every batch on the true optimal schedule for its
  own initial conditions, computed before the batch starts: the
  perfect-feedforward ceiling. It raises the mean and the best batches, but
  its minimum (4.57 g/L) is *worse* than the corrected policy's (5.72 g/L):
  a schedule fixed at time zero cannot answer a disturbance that develops
  while the batch runs. Feedforward adaptation and mid-course correction
  address the two different variance shares that
  :func:`~process_improve.simulation.variance_decomposition` separates.

The corrector's predictions were conservative: for the four corrected
batches it predicted 5.0 to 6.7 g/L and the executed titers came out 5.7 to
7.7 g/L. A latent-variable prediction regresses toward the mean, so a batch
deep in the tail is predicted less deep; the correction direction still
held.

Where to put the decision point
-------------------------------

Sweeping the single decision point over the batch (same seeds throughout,
mean executed gain over the batches corrected at that point):

=========  ====  ==============  ===============
Sample     Day   Corrected       Mean gain [g/L]
=========  ====  ==============  ===============
4          2.0   21 (8 harmed)   +0.32
6          3.0   8 (3 harmed)    +0.82
8          4.0   4 (0 harmed)    +2.02
10         5.0   6 (0 harmed)    +0.67
12         6.0   7 (7 harmed)    -0.05
14         7.0   7 (7 harmed)    -0.28
=========  ====  ==============  ===============

The window is real and it is in the middle of the batch. Too early, the
projections are still uncertain, so the dead band passes batches that did
not need correcting and the model misdirects some of them. Too late, the
biology has already decided: the remaining schedule has almost no leverage,
and small model errors turn corrections into damage. On this process the
useful window is days 4 to 5, just after the growth phase reveals which
batches are behind.

Practical notes
---------------

- The model is fitted on *recorded* (noisy, realised) trajectories but the
  corrector outputs *setpoints*. That is standard identification practice;
  it attenuates the apparent gain slightly, and the executed evaluation
  absorbs the difference.
- Setpoint bounds handed to the corrector are tightened inward by about two
  control-error standard deviations, so optimised schedules do not sit on
  actuator rails where clipping would bias the realised mean.
- The exploration weights (``weights["t2"]``, ``weights["spe"]``, and the
  hard caps) are the manufacturing-versus-development dial. On this
  configuration, relaxing the T2 penalty monotonically improved both the
  predicted and the executed gains of the corrected batches, because the
  poorest class's true optimum lies well outside the historical envelope;
  the harmed batches at late decision points show the same freedom working
  against you when the model's leverage is gone. There is no
  one-size-fits-all setting: measure it, on data the model has never seen.

Agent tools ``correct_batch_midcourse`` and
``evaluate_batch_control_policy``, and the ``midcourse_correction`` recipe,
expose this workflow to agent callers.

.. note::

   The accompanying book, `Process Improvement using Data
   <https://learnche.org/pid>`_, develops the full argument: golden-batch
   monitoring, prediction of a running batch, and the correction workflow,
   with these executed numbers.
