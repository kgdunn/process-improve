Batch PLS fault diagnosis: the simulated SBR reactor
====================================================

Case study for `issue #156 <https://github.com/kgdunn/process-improve/issues/156>`_.

Styrene-butadiene rubber (SBR) is made by emulsion polymerization in a batch
reactor. Six trajectories are recorded during each batch (reactor, cooling
water and jacket temperatures, latex density, conversion and the energy
released), and five quality attributes of the latex are measured at the end
(composition, particle size, branching, cross-linking and polydispersity).
The 53 batches were simulated from a first-principles model, which makes this
a rare kind of case study: the fault is known. Batches 34 and 37 both received
30% more organic impurity in the butadiene feed, from the very start of batch
37 and partway through batch 34.

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

Batch 37 sits at the low end of :math:`t_1` because its conversion and latex
density were below average, and the full contribution plot shows this for the
whole length of the batch. The raw overlays confirm it: the two trajectories
run under the others from the first sample. The impurity slowed the reaction
from the start, which is exactly the injected fault.

Batch 34: the same fault, partway through
-----------------------------------------

.. literalinclude:: sbr_batch_pls.py
   :language: python
   :start-after: # -- section: batch-34 --
   :end-before: # -- end: batch-34 --

.. code-block:: text

   batch 34: t2 contributions per tag = Conversion +8.1, CoolingTemp +12.2, EnergyReleased +14.3,
                                       JacketTemp +12.3, LatexDensity +7.2, ReactorTemp +4.0
   batch 34: 5% of the t2 contribution has accumulated by sample 38

Batch 34 is high on :math:`t_2`, and the contributions come from the energy
released, the jacket temperature and the cooling-water temperature. The full
contribution plot is flat for the first third of the batch and grows after
it; :func:`process_improve.batch.contribution_at_time_plot` at one sample
after the onset shows the same three tags. The raw overlays show the cooling
water and jacket temperatures departing from the other batches partway
through. The same impurity, injected later, first shows up in the heat
balance rather than in the conversion.

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
* Paul Nomikos and John F. MacGregor, "Multi-way partial least squares in
  monitoring batch processes", *Chemometrics and Intelligent Laboratory
  Systems*, **30**, 97-108, 1995.
* Kevin Dunn, *Latent Variable Methods* course notes (ConnectMV, 2011-2012),
  the SBR batch PLS example, CC BY-SA 3.0.
