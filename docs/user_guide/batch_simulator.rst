The batch (bioreactor) simulator
================================

A "golden batch" is the run where everything went right, and batch automation
commonly responds by replaying its charging recipe and setpoint schedules for
every future batch. Replaying is open-loop control: it assumes the
disturbances present during the golden run will recur, and they do not. The
:mod:`process_improve.simulation.batch` module exists to make the
consequences of that assumption measurable, with numbers that reproduce
exactly from a seed:

1. Replaying a golden schedule does not reproduce the golden outcome.
2. The spread persists even when the measured initial conditions are held
   identical, because disturbances also arise while the batch runs.
3. The quality variance under replay therefore splits into a share that
   adapting the schedule *before* the batch could address, a share observable
   only *during* the batch (the case for mid-course correction), and a noise
   floor.

Because the simulator's own kinetics are known, the *true* optimal schedule
for any batch's initial conditions is computable. That gives later,
data-driven schemes (latent-variable models, trajectory optimisation,
mid-course correction) a ceiling to be scored against, which no historical
dataset can offer.

The process model
-----------------

A 10-day fed-batch bioreactor, sampled twice daily: biomass, substrate,
product (titer) and volume, with pH and temperature as the manipulated
schedule and dissolved oxygen, offgas CO2 and volume as recorded responses.
Growth follows the gamma-concept cardinal temperature and pH model of Rosso
et al. (1995); production follows Luedeking and Piret (1959). Three couplings
give the schedule a real, condition-dependent optimum: an oxygen-transfer
ceiling with hypoxic death, strong hypothermic growth arrest (the reason the
industrial biphasic temperature shift works), and substrate consumption by
production with starvation death. The full equations, parameter meanings and
citations are in the module docstring
(:mod:`process_improve.simulation.batch`).

Three disturbance channels can each be scaled or switched off independently:
measured initial conditions (an 11-variable upstream ``Z`` block with three
feed classes A, B and C), an unmeasured within-batch disturbance that is
visible in the gas trajectories, and control-loop plus measurement noise at
instrument scale. All random draws are made on every call and multiplied by
their channel scale, so the same seed with one channel switched off is a true
counterfactual for the same batch.

Realism: the sensitivity budget
-------------------------------

A simulator whose quality output moves when an input moves by less than an
instrument can resolve is not believable. The nominal schedule therefore sits
at stationary points of the response (the pH optimum and the interior optimum
of the production-phase temperature hold), so instrument-scale deviations are
second order, while sustained multi-degree deviations cross real mechanisms
and cost real titer. Measured on the default configuration (nominal titer
8.01 g/L, 200 noise replicates, seed 0):

===============================================  =================
Perturbation                                     Effect on titer
===============================================  =================
Control-loop noise, sd 0.15 degC and 0.02 pH     sd 0.25%
Sustained bias of +0.1 / -0.1 degC               -0.22% / -0.20%
Sustained bias of +0.02 / -0.02 pH               -0.01%
One 12-hour sample at +0.5 degC                  -0.30%
Sustained bias of +1.0 / -1.0 degC               -11.5% / -7.1%
Sustained bias of +2.0 / -2.0 degC               -20.6% / -22.5%
===============================================  =================

These numbers are enforced as acceptance tests in
``tests/test_simulation_batch.py``, and
:meth:`~process_improve.simulation.batch.BioreactorSimulator.sensitivity_budget`
recomputes the table from the live configuration, so the claim can be checked
against any parameter changes rather than taken on trust:

.. code-block:: python

    from process_improve.simulation import BioreactorSimulator

    sim = BioreactorSimulator()
    print(sim.sensitivity_budget(random_state=0))

The three claims, in code
-------------------------

Replaying the schedule does not reproduce the outcome:

.. code-block:: python

    campaign = sim.simulate_campaign(50, policy="replay", random_state=0)
    titer = campaign.quality["titer"]
    print(titer.mean(), titer.std(ddof=1))   # a spread of roughly 14% CV

Holding the measured initial conditions identical does not remove the spread.
``dataclasses.replace`` derives a configuration with the initial-condition
channel off; what remains (roughly 8% CV, about twenty-five times the noise
floor) arose during the batches:

.. code-block:: python

    import dataclasses
    from process_improve.simulation import BioreactorConfig

    same_z = BioreactorSimulator(
        dataclasses.replace(BioreactorConfig(), ic_scale=0.0, noise_scale=0.0)
    )
    campaign = same_z.simulate_campaign(50, policy="replay", random_state=0)

And the decomposition, which runs the four campaigns and reports the
interaction residual of the nonlinear model explicitly instead of forcing the
buckets to sum:

.. code-block:: python

    from process_improve.simulation import variance_decomposition

    print(variance_decomposition(sim, n_batches=200, random_state=0))

The golden batch and the adaptation ceiling
-------------------------------------------

:meth:`~process_improve.simulation.batch.BioreactorSimulator.golden_trajectory`
finds the true optimal schedule for the *nominal* initial conditions by
optimising over the simulator's own kinetics; it recovers the industrial
biphasic shape (a warm growth phase near 37.5 degC, a smooth downshift into a
production hold near 29 degC) without being told it. On the defaults it
reaches 8.54 g/L against 8.01 g/L for the built-in nominal recipe.

:meth:`~process_improve.simulation.batch.BioreactorSimulator.optimal_trajectory`
does the same for any batch's own initial conditions. The gap between
replaying the golden schedule and each batch's own optimum is the value a
perfect feedforward adaptation could recover; on the default configuration it
ranges from under 1% for a good feed lot to roughly 20% for the poorest.
The ``"adapted"`` campaign policy runs every batch at its own optimum, as
that ceiling, and the ``"historical"`` policy adds deliberate setpoint
variation, since a perfectly consistent history carries no information about
how the controls affect quality.

.. code-block:: python

    golden = sim.golden_trajectory()
    adapted = sim.optimal_trajectory(campaign.initial_conditions.iloc[0])

Campaign outputs use the package's standard batch dictionary format
(``dict[batch_id, DataFrame]``), so they feed directly into
:func:`process_improve.batch.data_input.dict_to_wide` and the alignment
tooling for latent-variable modelling.

Agent tools
-----------

Two registered tools expose the baseline to agent callers:
``simulate_batch_campaign`` (campaign outcomes with the disturbance-free
reference titer for comparison) and ``decompose_batch_quality_variance`` (the
variance split). The ``golden_batch_baseline`` analysis recipe chains them
into the three-step argument above.

.. note::

   The accompanying book, `Process Improvement using Data
   <https://learnche.org/pid>`_, develops the golden-batch discussion and the
   latent-variable methods this baseline is built to support.
