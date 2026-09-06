Batch PLS fault diagnosis: the simulated SBR reactor
====================================================

Case study for `issue #156 <https://github.com/kgdunn/process-improve/issues/156>`_.

Styrene-butadiene rubber (SBR) is made by emulsion polymerization in a batch
reactor. Six trajectories are recorded during each batch (reactor, cooling
water and jacket temperatures, latex density, conversion and the energy
released), and five quality attributes of the latex are measured at the end
(composition, particle size, branching, cross-linking and polydispersity).
The 53 batches were simulated from a first-principles model, which makes this
a rare kind of case study: the fault is known (Nomikos and MacGregor, 1994).
Batch 37 received 30% more organic impurity in the butadiene feed than the
normal batches from its very start, and batch 34 50% more from midway
through.

The complete script is ``sbr_batch_pls.py`` in this directory:

.. code-block:: bash

   uv run python docs/user_guide/case_studies/batch/sbr_batch_pls.py --output-dir case-study-output/sbr

It prints the numbers quoted below and writes every figure as an HTML file to
the output directory.

Data
----

`SBR batch reactor <https://openmv.net/info/sbr-batch-reactor>`_: 53 batches
of 200 samples with nine trajectories, and five quality attributes per batch.
:func:`process_improve.batch.load_sbr` downloads the workbook and returns
the batch dictionary, the quality table, and the list of the six trajectories
the original study modelled. The two feed flows are constant in the simulation
and the feed temperature barely moves, so they are left out.

.. literalinclude:: sbr_batch_pls.py
   :language: python
   :start-after: # -- section: constants --
   :end-before: # -- end: constants --

.. literalinclude:: sbr_batch_pls.py
   :language: python
   :start-after: # -- section: load --
   :end-before: # -- end: load --

.. literalinclude:: sbr_batch_pls.py
   :language: python
   :start-after: # -- section: raw --
   :end-before: # -- end: raw --

The batch PLS model
-------------------

:class:`process_improve.batch.BatchPLS` unfolds each batch into one row of
6 tags times 200 samples, scales every column to unit variance, and fits a
PLS model from that row to the five quality attributes.

.. literalinclude:: sbr_batch_pls.py
   :language: python
   :start-after: # -- section: model --
   :end-before: # -- end: model --

.. code-block:: text

   R2X per component = 0.245, 0.127; R2Y per component = 0.653, 0.069
   lowest t1 = [37, 34, 38]; highest t2 = [34, 9]
   batch 34: T2 = 28.2 (limit 6.6), SPE = 23.1 (limit 34.6)
   batch 37: T2 = 19.2 (limit 6.6), SPE = 18.7 (limit 34.6)

The first component explains 24.5% of the trajectories and 65.3% of the
quality block; the second adds 12.7% and 6.9%. The score plot flags both
faulty batches, which is encouraging: batch 37 has the lowest :math:`t_1` of
all batches, and batch 34 the highest :math:`t_2`. Both are far outside the
Hotelling's :math:`T^2` limit. The SPE, on the other hand, flags neither.
The SPE of a whole batch averages the residuals over 200 samples, so a
deviation that the model can describe (a shift along the components) does
not show up there. SPE and scores answer different questions.

Where the model explains the trajectories
-----------------------------------------

Every unfolded column has its own :math:`R^2`, so the fit can be read per tag
and per time sample.

.. literalinclude:: sbr_batch_pls.py
   :language: python
   :start-after: # -- section: r2-breakdown --
   :end-before: # -- end: r2-breakdown --

.. code-block:: text

   R2 per tag, averaged over time: Conversion 0.75, CoolingTemp 0.23, EnergyReleased 0.26,
                                   JacketTemp 0.24, LatexDensity 0.67, ReactorTemp 0.08

Latex density and conversion are the trajectories the model uses most, and
:math:`R^2` is low at the start of every trajectory because all batches begin
alike. The time-varying weights :math:`w_1` and :math:`w_2`, drawn with
:func:`process_improve.batch.time_varying_loading_plot`, show the same
picture per component.

Batch 37: the fault from the start
----------------------------------

.. literalinclude:: sbr_batch_pls.py
   :language: python
   :start-after: # -- section: batch-37 --
   :end-before: # -- end: batch-37 --

.. code-block:: text

   batch 37: t1 contributions per tag = Conversion -33.4, CoolingTemp -4.2, EnergyReleased -5.2,
                                       JacketTemp -4.3, LatexDensity -25.5, ReactorTemp -1.3
   batch 37: share of the t1 contribution per fifth of the batch = 15%, 18%, 19%, 26%, 22%
   batch 37: first sustained departure from the other batches: {'ReactorTemp': None,
             'CoolingTemp': None, 'JacketTemp': None, 'LatexDensity': 13, 'Conversion': 9, 'EnergyReleased': None}

Batch 37 sits at the low end of :math:`t_1` because its conversion and latex
density were below average, and the contribution is spread over the whole
batch: every fifth of it carries between 15% and 26% of the total. The raw
data confirm it. ``sustained_departure`` expresses each trajectory of a faulty
batch as a distance from the mean of the normal batches, in units of their
standard deviation at that sample, and reports the first sample from which a
tag stays more than two standard deviations away for twenty samples in a row
(a single crossing is not informative, because a noisy tag crosses that line
now and then in every batch). Conversion and latex density of batch 37 depart
at samples 9 and 13 and run under the other batches to the end; none of its
other four trajectories stays outside the band for twenty samples. The
impurity slowed the reaction from the start, which is the injected fault.

Batch 34: the same fault, from the middle of the batch
------------------------------------------------------

.. literalinclude:: sbr_batch_pls.py
   :language: python
   :start-after: # -- section: batch-34 --
   :end-before: # -- end: batch-34 --

.. code-block:: text

   batch 34: t2 contributions per tag = Conversion +8.1, CoolingTemp +12.2, EnergyReleased +14.3,
                                       JacketTemp +12.3, LatexDensity +7.2, ReactorTemp +4.0
   batch 34: share of the t2 contribution per fifth of the batch = 6%, 9%, 17%, 39%, 29%
   batch 34: first sustained departure from the other batches: {'ReactorTemp': None,
             'CoolingTemp': 103, 'JacketTemp': 104, 'LatexDensity': 129, 'Conversion': 123, 'EnergyReleased': 105}

Batch 34 is high on :math:`t_2`, and the contributions come from the energy
released, the jacket temperature and the cooling-water temperature. The
timing is different from batch 37 in both views. The first two fifths of the
batch carry only 15% of the :math:`t_2` contribution and the last two fifths
carry 68%, and in the raw data the cooling-water temperature, the jacket
temperature and the energy released leave the band of the other batches at
samples 103 to 105, the middle of the batch, while conversion and latex
density only do so at samples 123 and 129. The same impurity, injected midway,
shows up first in the heat balance of the reactor and only afterwards in the
extent of reaction.
:func:`process_improve.batch.contribution_at_time_plot` at sample 120 shows
the same three tags carrying the deviation.

The same fault appears in two different places of the score plot because it
started at two different times. A batch model describes deviations in
(tag, time) cells, so the time of an event is part of its signature. This is
what makes batch models useful for diagnosis, and it is also why a library of
"known faults" in score space needs the onset time as a coordinate.

Predicted quality
-----------------

.. literalinclude:: sbr_batch_pls.py
   :language: python
   :start-after: # -- section: predictions --
   :end-before: # -- end: predictions --

.. code-block:: text

   quality of the faulty batches (rank 1 = lowest of 53 batches)
                              Composition  ParticleSize  Branching  CrossLinking  Polydispersity
   value            batch_id
   observed         34             0.4525          1244  1.234e-05     4.784e-05           3.599
                    37             0.4525          1247  1.173e-05     4.549e-05           3.462
   predicted        34              0.454          1245  1.228e-05     4.761e-05           3.577
                    37               0.45          1250  1.183e-05     4.585e-05           3.491
   rank of observed 34                  5             1          4             4              17
                    37                  4             2          1             1               1

Both batches produced poor latex: batch 37 has the lowest branching,
cross-linking and polydispersity of all 53 batches and batch 34 the smallest
particle size. The fitted values from the PLS model place them at the same
end of every attribute, so a quality prediction from the trajectories would
have flagged both batches before the laboratory did. Batch 37 is predicted
low on every attribute; batch 34 is predicted only mildly low on
polydispersity, because the :math:`t_2` direction that carries its fault
explains 6.9% of the quality block.

Predicting quality before the batch ends
----------------------------------------

The model above was fitted on complete batches, and a plant would like to
know the quality while the batch is still running. The unfolded row of a
running batch is complete up to the newest sample and missing after it, so
this is a missing-data problem: after :math:`k` samples, estimate the scores
from the cells observed so far, and read the quality prediction off the
scores through the model's Y loadings. The scores are estimated with trimmed
score regression (Arteaga and Ferrer, 2002), a regression of the scores on
the observed cells built from the training batches, the estimator that
Garcia-Munoz, Kourti and MacGregor (2004) found gives stable score estimates
from the first samples of a batch.
:meth:`process_improve.batch.BatchPLS.predict_online` does this for one
point in time, and
:meth:`process_improve.batch.BatchPLS.predict_online_trace` for every sample
of a complete batch, as the prediction would have evolved in real time.

.. literalinclude:: sbr_batch_pls.py
   :language: python
   :start-after: # -- section: online-prediction --
   :end-before: # -- end: online-prediction --

.. code-block:: text

   batch 4: ParticleSize measured 1256.9, fitted from the complete batch 1257.1
   batch 4: ParticleSize predicted after 10 samples 1251.0, 25 samples 1248.1, 50 samples 1255.3,
            100 samples 1254.9, 150 samples 1257.4, 200 samples 1257.1
   RMSEE / sd of ParticleSize after 10 samples 2.84, 50 samples 1.40, 100 samples 1.29,
                                  150 samples 0.62, 200 samples 0.60
   RMSEE / sd of Branching after 10 samples 2.10, 50 samples 0.88, 100 samples 0.81,
                               150 samples 0.32, 200 samples 0.24

Batch 4 is the batch whose trajectories lie closest to the average. Its
particle size is predicted 6 to 9 units away from the measured value in the
first 25 samples and within about 2 units from sample 50 onwards, and after
200 samples the prediction equals the fitted value from the complete batch,
as it must, because the row is then complete. One batch says little about
the error, though. :meth:`process_improve.batch.BatchPLS.online_rmse` traces
every training batch and pools the squared errors sample by sample; on the
training batches this is the root-mean-square error of estimation, RMSEE, as
a function of how much of the batch has been observed, and its last value is
the RMSEE of the model fitted on complete batches. Dividing by the standard
deviation of each attribute puts the five attributes on one axis, where a
ratio of about 1 is the error of predicting the average batch every time.

For particle size the ratio is 2.84 after 10 samples, still 1.29 at the
halfway point and 0.62 after 150 samples: in the first half of the batch the
prediction is no better than the average batch, and it improves in the
second half. Branching, whose final RMSEE is a quarter of its standard
deviation, is below 1 by sample 50. The early predictions are worse than the
average because few cells have been observed, and they are the cells where
every batch begins alike (the :math:`R^2` breakdown showed this), so the
score estimate carries little information about the batch. The RMSEE is
measured on the batches the model was fitted to. Refitting the model with
one batch left out and tracing that batch, which the script leaves out
because it refits 53 models, gives a prediction error, RMSEP, of 3.10, 1.66,
1.65, 0.84 and 0.78 standard deviations at the same five samples: the same
shape, at a higher level.

Would the model have caught it on-line?
---------------------------------------

The score plot flagged both faulty batches once they were complete. The
question here is whether a chart could have flagged them while they were
running, and after how many samples. Two things change from the model
above. A reference model must describe normal operation, so batches 34 and
37 are left out and a two-component model is refitted on the other 51
batches. And a running batch needs a limit at every sample rather than one
limit for the whole batch: the score estimates early in a batch are shrunk
and noisy compared with those near its end, so the scatter of the reference
batches differs from sample to sample.
:class:`process_improve.batch.BatchMonitor` passes every reference batch
through ``predict_online_trace`` and summarises the spread at each sample,
following Nomikos and MacGregor (1995), who set the limits of their score
charts from that spread and note that the :math:`T^2` chart needs the score
covariance at each sample as well. The :math:`T^2` at sample :math:`k`
is standardised by the covariance of the reference batches' score estimates
at that sample, as Garcia-Munoz, Kourti and MacGregor (2004) compute it,
which gives one limit for the whole batch, and the SPE
limit is a chi-squared limit fitted to the reference batches' SPE at that
sample (``spe_window`` pools the values of neighbouring samples into that
fit, which steadies the limits when few reference batches are available;
on these 51 batches a window of two samples either side leaves every alarm
sample unchanged). The SPE charted is the instantaneous one, the residual of the newest
sample only, which reacts in the sample a fault begins; the cumulative SPE
over every cell observed so far is diluted by the earlier, normal samples,
and here it reacts to batch 34 after 112 samples instead of 105.
:func:`process_improve.batch.online_monitoring_plot` draws either chart.

.. literalinclude:: sbr_batch_pls.py
   :language: python
   :start-after: # -- section: online-monitoring --
   :end-before: # -- end: online-monitoring --

.. code-block:: text

   reference model on 51 batches; T2 limit 10.54 at every sample
   batch 34: first 3 consecutive samples above the limit: T2 after 190 samples,
             SPE after 105 samples
   batch 37: first 3 consecutive samples above the limit: T2 after 23 samples,
             SPE after 145 samples
   normal batches: 0.17% of the T2 values and 1.24% of the SPE values above their limits
   batch 34: share of the SPE after 105 samples per tag = ReactorTemp 40%, CoolingTemp 30%,
             JacketTemp 16%, EnergyReleased 11%, Conversion 2%, LatexDensity 1%
   batch 34: share of the SPE after 109 samples per tag = CoolingTemp 36%, JacketTemp 29%,
             EnergyReleased 19%, ReactorTemp 12%, Conversion 2%, LatexDensity 1%
   batch 37 Conversion, mean of the samples after 30: forecast 0.6472, actual 0.6420,
                        average of the normal batches 0.6609
   batch 37 Conversion, mean of the samples after 60: forecast 0.6480, actual 0.6477,
                        average of the normal batches 0.6656
   batch 34 CoolingTemp, mean of the samples after 60: forecast 46.6357, actual 46.7744,
                         average of the normal batches 46.6152
   batch 34 CoolingTemp, mean of the samples after 115: forecast 46.6366, actual 46.8173,
                         average of the normal batches 46.6015

An alarm counts once the statistic stays above its 99% limit for three
consecutive samples. A single crossing is not informative: the normal
batches put 0.17% of their :math:`T^2` values and 1.24% of their SPE values
above the limits, so a batch of 200 samples crosses the SPE limit now and
then (batch 34 has one isolated crossing after 37 samples, and batch 4, the
near-average batch, one after 26).

Batch 37 shows in the :math:`T^2` chart after 23 samples: its :math:`T^2`
is 9.6 after 21 samples, 10.2 after 22, 10.7 after 23 and 11.9 after 25,
against a limit of 10.54, and it stays above the limit for the rest of the
batch. The impurity slowed the reaction from the first sample, and low
conversion and latex density is the direction the reference model's first
component describes, so the score estimate moves along :math:`t_1` as soon
as enough samples have been observed to estimate it. The instantaneous SPE
of batch 37 does not react until sample 145; the deviation lies in the model
plane and leaves little residual.

Batch 34 shows in the SPE chart after 105 samples, within a couple of
samples of the point where the cooling-water temperature, the jacket
temperature and the energy released leave the band of the other batches in
the raw data (samples 103 to 105). At the alarm sample the reactor
temperature carries 40% of the squared residual and the cooling-water
temperature 30%; four samples later the cooling-water temperature, the
jacket temperature and the energy released carry 84% between them, the same
three tags the whole-batch contribution plot named. The :math:`T^2` of
batch 34 does not alarm until sample 190. None of the reference batches has
a heat-balance deviation that starts midway, so the reference model has no
component for it; the deviation is off the model plane and shows in the
residual, not in the scores. The two charts answer different questions: the
:math:`T^2` reacts to a deviation along the components the reference
batches exhibited, the SPE to a direction the reference model never saw,
and which chart catches a fault first depends on the fault.

The same distinction decides whether the model can forecast the rest of a
running batch. ``predict_online`` also returns a ``forecast``: the batch's
own values up to the newest sample, and beyond it the trajectories implied
by the score estimate, Eq. 4 of Wold, Kettaneh-Wold, MacGregor and Dunn
(2009). For batch 37 the forecast from 30 samples puts the mean conversion
of the remaining samples at 0.6472, against 0.6420 observed and 0.6609 for
the average normal batch; from 60 samples the forecast is 0.6480 against
0.6477 observed. The scores had picked up the slow reaction, and the
forecast follows it. For batch 34 the forecast of the cooling-water
temperature stays at the average of the normal batches, 46.64 from 60
samples and again from 115 samples, while the batch ran at 46.77 and 46.82.
The fault of batch 34 does not move the scores of the reference model, and a
forecast made from the scores cannot see it.

Running the script
------------------

.. literalinclude:: sbr_batch_pls.py
   :language: python
   :start-after: def main(
   :end-before: if __name__ == "__main__":

References
----------

* Paul Nomikos, *Statistical process control of batch processes*, PhD thesis,
  McMaster University, 1995.
* Paul Nomikos and John F. MacGregor, "Monitoring batch processes using
  multiway principal component analysis", *AIChE Journal*, **40**, 1361-1375,
  1994, https://literature.learnche.org/item/30/monitoring-batch-processes-using-multiway-principal-component-analysis
  The simulation and the two faulty batches.
* Paul Nomikos and John F. MacGregor, "Multi-way partial least squares in
  monitoring batch processes", *Chemometrics and Intelligent Laboratory
  Systems*, **30**, 97-108, 1995,
  https://literature.learnche.org/item/32/multi-way-partial-least-squares-in-monitoring-batch-processes
* Paul Nomikos and John F. MacGregor, "Multivariate SPC charts for monitoring
  batch processes", *Technometrics*, **37**, 41-59, 1995,
  https://literature.learnche.org/item/34/multivariate-spc-charts-for-monitoring-batch-processes
* Francisco Arteaga and Alberto Ferrer, "Dealing with missing data in MSPC:
  several methods, different interpretations, some examples", *Journal of
  Chemometrics*, **16**, 408-418, 2002,
  https://literature.learnche.org/item/20/dealing-with-missing-data-in-mspc-several-methods-different-interpretations-some-examples
* Salvador Garcia-Munoz, Theodora Kourti and John F. MacGregor, "Model
  predictive monitoring for batch processes", *Industrial & Engineering
  Chemistry Research*, **43**, 5929-5941, 2004,
  https://literature.learnche.org/item/157/model-predictive-monitoring-for-batch-processes
* Svante Wold, Nouna Kettaneh-Wold, John F. MacGregor and Kevin G. Dunn,
  "Batch process modeling and MSPC", *Comprehensive Chemometrics*, chapter
  2.10, 163-197, 2009,
  https://literature.learnche.org/item/155/batch-process-modeling-and-mspc
* Kevin Dunn, *Latent Variable Methods* course notes (ConnectMV, 2011-2012),
  the SBR batch PLS example, CC BY-SA 3.0.
