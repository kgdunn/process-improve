Multiblock batch PLS: the FMC batch dryer
=========================================

Case study for `issue #154 <https://github.com/kgdunn/process-improve/issues/154>`_.

An agricultural chemical is dried in an industrial batch dryer. Wet cake, the
solid with its embedded solvent, is charged to the dryer and dried through
three recipe phases: the solvent is collected in a side tank, the temperature
is ramped, and the batch is cooled down. Chemical changes take place in the
solid during drying, and the operators can adjust a few set points. Ten
trajectories are recorded over each batch, and three more blocks describe
each batch with one row: the chemistry of the cake before the batch
(``Zchem``, eleven measurements), the operating conditions and recipe timings
(``Zop``, nine values), and eight final quality attributes (``Y``). This is
the multiblock case study of Garcia-Munoz and co-workers (2003).

The questions are the ones a plant asks in this order: what does product
quality look like, do the initial conditions explain it, what do the
trajectories add, and which batches deserve a closer look. The original course
material answers them with a ladder of models, two components each, and this
script follows the same ladder.

The complete script is ``fmc_multiblock_batch_pls.py`` in this directory:

.. code-block:: bash

   uv run python docs/user_guide/case_studies/batch/fmc_multiblock_batch_pls.py --output-dir case-study-output/fmc

It prints the numbers quoted below and writes every figure as an HTML file to
the output directory.

Data
----

`Industrial batch dryer <https://openmv.net/info/batch-dryer>`_: a workbook
with the four blocks over 59 batches. The trajectories were aligned within
each of the three phases before the data were archived, to 325 samples per
batch, and ``ClockTime``, the wall time at each aligned sample, is carried
along as an eleventh trajectory: after alignment it is no longer a clock but a
record of how much each batch was stretched or compressed, which is itself
information about the batch. (For raw, unaligned data see
:func:`process_improve.batch.batch_dtw`; the unaligned trajectories of this
same process are bundled as :func:`process_improve.batch.load_dryer`.)

Thirteen batches have no chemistry measurements. The original study excluded
them, and :func:`process_improve.batch.load_fmc` returns their identifiers as
``missing_chemistry`` so the exclusion can be reproduced. The remaining 46
batches still contain genuine missing cells: 19 in the quality block, one in
the chemistry block, and 1220 in the trajectories of ten batches. The
:mod:`process_improve.multivariate` estimators handle missing values through
their NIPALS path, which is why this case study uses :class:`~process_improve.multivariate.PCA`,
:class:`~process_improve.multivariate.PLS` and
:class:`~process_improve.multivariate.methods.MBPLS` directly; the batch
classes :class:`~process_improve.batch.BatchPCA` and
:class:`~process_improve.batch.BatchPLS` require complete data.

.. literalinclude:: fmc_multiblock_batch_pls.py
   :language: python
   :start-after: # -- section: constants --
   :end-before: # -- end: constants --

.. literalinclude:: fmc_multiblock_batch_pls.py
   :language: python
   :start-after: # -- section: load --
   :end-before: # -- end: load --

.. code-block:: text

   46 batches kept; missing cells: Y 19, Zchem 1, X in batches [20, 22, 27, 28, 31, 55, 60, 61, 67, 71]

.. literalinclude:: fmc_multiblock_batch_pls.py
   :language: python
   :start-after: # -- section: raw --
   :end-before: # -- end: raw --

Characterising product quality
------------------------------

A PCA on the eight quality attributes shows how the batches group in quality
space before any process data are involved.

.. literalinclude:: fmc_multiblock_batch_pls.py
   :language: python
   :start-after: # -- section: quality --
   :end-before: # -- end: quality --

.. code-block:: text

   PCA on Y: R2 cumulative = 0.500, 0.703
   batch 61: t1 contributions = Y1 -0.76, Y2 missing, Y4 -1.04, Y6 -1.05, Y9 -0.10, Y10 -0.77, Y11 -1.07, SolventConc +0.02
   batch 14: t1 contributions = Y1 +0.77, Y2 +0.28, Y4 +0.73, Y6 +0.78, Y9 -0.04, Y10 +0.84, Y11 +0.15, SolventConc -0.11

Two components explain 70% of the quality block and the score plot shows two
groups of batches. Batches 61 and 14 are one member of each group; their
:math:`t_1` contributions are mirror images, with the same attributes
(``Y1``, ``Y4``, ``Y6``, ``Y10`` and ``Y11``) low in one group and high in
the other, so the first component is a general quality level rather than a
trade-off between attributes. A missing quality cell simply has no
contribution.

Effect of the initial conditions
--------------------------------

.. literalinclude:: fmc_multiblock_batch_pls.py
   :language: python
   :start-after: # -- section: initial-conditions --
   :end-before: # -- end: initial-conditions --

.. code-block:: text

   PLS Zchem -> Y: R2Y cumulative = 0.163, 0.222
   PLS Zop -> Y: R2Y cumulative = 0.207, 0.262
   batch 20 on Zop: t1 contributions = Level1 -0.33, Temp1 +0.02, Temp2 -0.02, Time4 -1.01, Time1 -0.14,
                                       Time2 -1.42, Time3 -0.15, TempSlope -1.27, WgtCake +0.19

Each initial-condition block on its own explains about a quarter of the
quality block. Batch 20 stands out in the operating-condition model through
its recipe timings (``Time2``, ``Time4``) and temperature slope, which is
worth remembering when it turns up again in the trajectory models.

Multiblock PLS on the initial conditions
----------------------------------------

The two blocks can be modelled together. :class:`~process_improve.multivariate.methods.MBPLS`
scales each block on its own and then weights it by :math:`1/\sqrt{K_b}`, so
the eleven chemistry columns and the nine operating columns pull on the
super-score with equal total weight; the block scores show what each block
contributes, and the super-score plot shows the batches in the combined
space.

.. literalinclude:: fmc_multiblock_batch_pls.py
   :language: python
   :start-after: # -- section: multiblock-z --
   :end-before: # -- end: multiblock-z --

.. code-block:: text

   MBPLS Z -> Y: R2Y cumulative = 0.292, 0.364; R2X per block after 2 components = Zchem 0.296, Zop 0.356
   batch 20, block Zchem: t1 contributions = Z1 -0.08, Z2 +0.03, Z3 -0.00, Z4 +0.04, Z5 +0.00, Z6 +0.03, Z7 -0.05, ...
   batch 20, block Zop: t1 contributions = Level1 -0.08, Temp1 +0.02, Temp2 -0.01, Time4 -0.25, Time1 -0.04, Time2 -0.38, ...

Together the blocks explain 36% of the quality block, more than either alone,
and the per-block contributions show that batch 20 is unusual in its
operating conditions, not in its chemistry.

The trajectories alone
----------------------

The trajectories are unfolded batchwise with
:func:`process_improve.batch.dict_to_wide`: one row per batch of 10 tags times
325 samples. ``MCUVScaler`` returns flat column labels, so the 2-level
``(tag, sequence)`` index is re-attached after scaling; the batch plots read
it.

.. literalinclude:: fmc_multiblock_batch_pls.py
   :language: python
   :start-after: # -- section: unfold --
   :end-before: # -- end: unfold --

.. literalinclude:: fmc_multiblock_batch_pls.py
   :language: python
   :start-after: # -- section: batch-pca --
   :end-before: # -- end: batch-pca --

.. code-block:: text

   unfolded trajectories: 46 batches x 3250 columns, 1220 missing cells
   batch PCA on X: R2 cumulative = 0.197, 0.349
   largest SPE among the complete batches: batch 51; share per tag = Agitator 5%, CTankLvl 0%, D-Temp 12%,
       D-Temp-SP 27%, DiffPres 1%, DryPress 6%, J-Temp 24%, J-Temp-SP 18%, Power 3%, Torque 4%

Two components describe 35% of the batch-to-batch variation in the
trajectories, and the time-varying loading plot shows where in the batch each
component acts. Contribution plots are only defined for batches with complete
trajectories, so batch 20 is examined through its raw overlays (dryer
temperature, power and torque), while the SPE contributions of batch 51, the
complete batch with the largest SPE, point at the dryer and jacket temperature
set points and the jacket temperature.

Trajectories to quality
-----------------------

.. literalinclude:: fmc_multiblock_batch_pls.py
   :language: python
   :start-after: # -- section: batch-pls --
   :end-before: # -- end: batch-pls --

.. code-block:: text

   batch PLS X -> Y: R2Y cumulative = 0.274, 0.392
   batch 13: t1 contributions per tag = Agitator -2.0, CTankLvl -9.0, D-Temp -5.3, D-Temp-SP -2.3, DiffPres -1.5,
             DryPress -1.2, J-Temp -1.5, J-Temp-SP -4.8, Power -3.5, Torque -3.3

The trajectories explain 39% of the quality block, more than the initial
conditions did. Batch 13 is at one end of :math:`t_1`, and its contributions
are spread over the tags with the collector tank level (the solvent removal)
and the dryer temperature leading; batches 5 and 7 are examined the same way.
The observed-versus-predicted plot of ``SolventConc`` shows how well the
residual solvent, the attribute the plant cares most about, follows from the
trajectories.

Batch multiblock PLS
--------------------

The final model joins all three X blocks. The trajectory block enters as
3250 columns, and its :math:`1/\sqrt{K_b}` weight keeps it from drowning out
the two small blocks.

.. literalinclude:: fmc_multiblock_batch_pls.py
   :language: python
   :start-after: # -- section: batch-mbpls --
   :end-before: # -- end: batch-mbpls --

.. code-block:: text

   batch MBPLS: R2Y cumulative = 0.375, 0.468; R2X per block after 2 components = Zchem 0.243, Zop 0.308, X 0.217
   super VIP per block: Zchem 0.88, Zop 1.10, X 1.00

The combined model explains 47% of the quality block, and the super VIP
ranks the operating conditions first, the trajectories second and the
chemistry last. This is the model to build the stagewise monitoring and the
final-quality prediction on: the super-score plot places every batch in one
space, the block scores say whether a batch is unusual in its chemistry, its
operation or its trajectories, and the X-block contributions of a batch, drawn
with :func:`process_improve.batch.unfolded_contribution_plot`, name the tags
and the phase.

Where to go next
----------------

The course notes suggest replacing the raw trajectories with feature blocks
(timings, temperatures, impeller and pressure summaries) so that the model
can be read by phase, and building the online monitoring model with
:class:`process_improve.batch.BatchMonitor` once the missing cells have been
filled in or the incomplete batches removed.

Running the script
------------------

.. literalinclude:: fmc_multiblock_batch_pls.py
   :language: python
   :start-after: def main(
   :end-before: if __name__ == "__main__":

References
----------

* Salvador Garcia-Munoz, Theodora Kourti, John F. MacGregor, Antonio G. Mateos
  and Gerry Murphy, "Troubleshooting of an industrial batch process using
  multivariate methods", *Industrial and Engineering Chemistry Research*,
  **42**, 3592-3601, 2003.
* Svante Wold, Nouna Kettaneh-Wold, John F. MacGregor and Kevin G. Dunn,
  "Batch process modeling and MSPC", *Comprehensive Chemometrics*, **2.10**,
  163-197, 2009.
* Salvador Garcia-Munoz, *Batch process improvement using latent variable
  methods*, PhD thesis, McMaster University, 2004.
* Kevin Dunn, *Latent Variable Methods* course notes (ConnectMV, 2011-2012),
  the FMC multiblock batch PLS example, CC BY-SA 3.0.
