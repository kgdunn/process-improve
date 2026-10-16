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

Every gain on this page is *executed*: the corrected schedule is fed back
into the simulator with the identical disturbance seed, so each gain is a
same-batch counterfactual, not the model's own prediction. On an operating
plant a batch runs once: its corrected quality is measured, but the quality
it would have reached without the correction is not, so the gain on a single
plant batch rests on the model's no-change prediction. The simulator removes
that limit.

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
  not corrected; Flores-Cerrillo and MacGregor, 2004, check the SPE before
  correcting and suggest keeping the previous schedule when it fails), the
  no-correction dead band (correct only when the projected shortfall is
  significant against that interval, in the role of the no-control region of
  Yabuki and MacGregor, 1997; the default of 1.0 asks that the whole interval
  fall short of the target), and per-decision-point limits built the same way
  (Garcia-Munoz, Kourti and MacGregor, 2004).
- :func:`~process_improve.batch.control.evaluate_control_policies` runs the
  whole comparison end to end.

Two findings from building this page shape the defaults. The historical
campaign must contain deliberate setpoint moves in the *shapes* the
controller will use: the simulator's ``historical`` policy therefore varies
its schedules with independent knot offsets, and corrections use the same
knot basis. And a single global linear model averages a gain direction that
depends on the feed class (a warm hold in mid-batch helps a slow batch and,
on average, costs a fast one; with one global model the comparison below
corrects five batches for about a third of the per-class models' mean gain),
so the evaluation fits one model per feed class and
assigns a fresh batch to the nearest class centroid in standardised Z; with
the class ranges overlapping along the feed-quality axis that assignment is
right about 85% of the time, and a miss hands the batch the neighbouring
range's model.

The executed comparison
-----------------------

.. code-block:: python

    from process_improve.batch.control import evaluate_control_policies
    from process_improve.simulation import BioreactorSimulator

    result = evaluate_control_policies(BioreactorSimulator(), y_target=8.0, random_state=0)
    print(result.summary.round(3))

Measured on the default configuration (200 historical training batches, 40
fresh test batches, one decision point at sample 8, which is day 4 of 10;
seed 0; about seven minutes, dominated by the two ceiling policies):

===============  ======  ======  ======  ======
Policy           Mean    Sd      Min     Max
===============  ======  ======  ======  ======
replay           7.507   1.198   3.655   8.925
midcourse        7.786   0.750   5.652   8.925
oracle-from-k    7.984   0.509   6.392   8.925
adapted          7.824   1.013   4.573   9.309
===============  ======  ======  ======  ======

Eight of the forty batches were corrected, six in the poorest feed class and
two in the middle one; thirty-one were left alone by the dead band and one was
stopped by the SPE validity gate. Reading the table:

- The corrected batches gained between +0.48 and +2.45 g/L, mean +1.40 g/L,
  and none was harmed. The worst batch in the campaign rose from 3.66 to
  5.65 g/L.
- The campaign standard deviation fell from 1.20 to 0.75 g/L, a 37%
  reduction, with the mean up 0.28 g/L: mid-course correction works on the
  low tail, which is where the money is when the target is a floor.
- The **oracle-from-k** row re-optimises the remaining schedule of those
  same corrected batches against the simulator itself (the true process) at
  the same decision point, with each batch's own seed: an estimate of the
  ceiling for any mid-course scheme there, which also knows the disturbances
  still to come. The data-driven correction captured 58% of the oracle's
  mean improvement; the rest mixes the error of an empirical model confined
  to the region its history explored, that foresight, and the corrector's
  own limits (movement penalty, validity caps, tighter bounds and rate
  limits).
- The **adapted** row runs every batch on the schedule that maximises the
  disturbance-free titer for its own initial conditions, computed before the
  batch starts from the simulator's own equations: an estimate of the
  feedforward ceiling. It raises the mean and the best batches, but
  its minimum (4.57 g/L) is *worse* than the corrected policy's (5.65 g/L):
  a schedule fixed at time zero cannot answer a disturbance that develops
  while the batch runs. Feedforward adaptation and mid-course correction
  address the two different variance shares that
  :func:`~process_improve.simulation.variance_decomposition` separates.

The corrector's predictions were conservative: for each of the eight
corrected batches the predicted gain was smaller than the executed one.

Where to put the decision point
-------------------------------

Sweeping the single decision point over the batch (same seeds throughout;
mean gain over the batches corrected at that point, as the model predicted
it, as executed, and for the oracle from the same point on the same
batches):

=========  ====  ==============  ===============  ==============  ============
Sample     Day   Corrected       Predicted [g/L]  Executed [g/L]  Oracle [g/L]
=========  ====  ==============  ===============  ==============  ============
4          2.0   9 (2 harmed)    +1.43            +1.12           +1.96
6          3.0   9 (1 harmed)    +1.14            +1.42           +2.19
8          4.0   8 (0 harmed)    +0.68            +1.40           +2.39
10         5.0   8 (0 harmed)    +0.19            +0.49           +2.07
12         6.0   8 (7 harmed)    +0.04            -0.03           +1.86
14         7.0   8 (8 harmed)    +0.09            -0.23           +1.60
=========  ====  ==============  ===============  ==============  ============

The window is real and it is in the middle of the batch. Too early, the
prediction has not yet separated the batches that will fall short from those
that will not: at day 2 the dead band admits nine batches, two of them are
harmed, and the mean gain is below the values at days 3 and 4. Too late, the
process still responds to its schedule (the oracle gains 1.6 g/L on the same
batches even from day 7), but the model no longer sees it: its predicted
gains fall toward zero, and the corrections computed from them turn into
damage. On this process the useful window is days 3 to 5. Days 3 and 4 give
the same mean gain, about +1.4 g/L, just after the growth phase reveals which
batches are behind, and day 3 harms one batch where day 4 harms none.

Practical notes
---------------

- The model is fitted on *recorded* trajectories (the realised values plus
  measurement noise), but the corrector outputs *setpoints*. The
  substitution is sound where the control loops track their setpoints, and
  the executed evaluation measures the realised effect either way.
- Setpoint bounds handed to the corrector are tightened inward by about two
  control-error standard deviations, so optimised schedules do not sit on
  actuator rails where clipping would bias the realised mean.
- The exploration weights (``weights["t2"]``, ``weights["spe"]``, and the
  hard caps) are the manufacturing-versus-development dial. On this
  configuration, relaxing the T2 penalty monotonically improved both the
  predicted and the executed gains of the corrected batches, because the
  poorest class's best schedules lie far from the centre of the historical
  data, where the T2 penalty holds the correction back; the harmed batches at
  late decision points show the same freedom working against you when the
  model's estimate of the leverage is wrong. There is no
  one-size-fits-all setting: measure it, on data the model has never seen.

Agent tools ``correct_batch_midcourse`` and
``evaluate_batch_control_policy``, and the ``midcourse_correction`` recipe,
expose this workflow to agent callers.

.. note::

   The accompanying book, `Process Improvement using Data
   <https://learnche.org/pid>`_, develops the full argument: golden-batch
   monitoring, prediction of a running batch, and the correction workflow,
   with these executed numbers.
