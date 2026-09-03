Batch PCA outlier diagnosis: the DuPont polymerization reactor
==============================================================

Case study for `issue #155 <https://github.com/kgdunn/process-improve/issues/155>`_.

Industrial nylon is made in a batch autoclave. The critical quality of a batch
is measured in the laboratory about twelve hours after the batch has ended, so
there is no feedback that could correct a running batch, and a long hold-up
before its disposition is known. What the plant does have is the trajectory of
ten process measurements (temperatures, pressures and flows) over every batch.
The question of this case study is what those trajectories can tell about a
batch, and what they cannot.

The data were supplied by DuPont and are the worked example of Nomikos and
MacGregor (1995). Batches 40, 41, 42, 50, 51, 53, 54 and 55 had a final
quality well outside the acceptable limit; batches 38, 45, 46, 49 and 52 were
above or very close to it.

The complete script is ``dupont_batch_pca.py`` in this directory:

.. code-block:: bash

   uv run python docs/user_guide/case_studies/batch/dupont_batch_pca.py --output-dir case-study-output/dupont

It prints the numbers quoted below and writes every figure as an HTML file to
the output directory. The sections below quote it piece by piece.

Data
----

`Industrial batch polymerization <https://openmv.net/info/polymerization>`_:
55 batches, each already aligned to 100 equal time intervals, with ten tags
per interval. The values are scaled for confidentiality.
:func:`process_improve.batch.load_dupont` downloads the file and returns the
standard batch dictionary, one 100-by-10 frame per batch.

.. literalinclude:: dupont_batch_pca.py
   :language: python
   :start-after: # -- section: constants --
   :end-before: # -- end: constants --

.. literalinclude:: dupont_batch_pca.py
   :language: python
   :start-after: # -- section: load --
   :end-before: # -- end: load --

Looking at one tag across all batches is the first check. The trajectories
overlay well (the alignment has already been done), and a few batches are
visibly unusual, but a plot per tag cannot rank 55 batches on ten variables
at once.

.. literalinclude:: dupont_batch_pca.py
   :language: python
   :start-after: # -- section: raw --
   :end-before: # -- end: raw --

Model A: batch PCA on all 55 batches
------------------------------------

:class:`process_improve.batch.BatchPCA` unfolds the batches batchwise: each
batch becomes one row of 10 tags times 100 samples, 1000 columns. Centring
the columns removes the average trajectory, and scaling them to unit variance
gives every (tag, time) cell the same weight, so the two components describe
how the batches deviate from the average batch.

.. literalinclude:: dupont_batch_pca.py
   :language: python
   :start-after: # -- section: model-a --
   :end-before: # -- end: model-a --

.. code-block:: text

   Model A: R2 per component = 0.383, 0.176; cumulative = 0.559
   Model A: largest |t1| = [54, 52, 50, 51]; largest |t2| = [53, 55, 50]
   Model A: largest SPE = batch 49 (39.3 vs 95% limit 29.1)

Two components explain 55.9% of the variance in the unfolded matrix. The
score plot shows batches 50 to 55 far from the rest; they pull the model
towards themselves, which is the first sign that the model needs to be rebuilt
without them. The SPE plot flags a different batch, 49, which sits in the
score plot among the normal batches: its problem is not a large deviation
along the main directions of variation but a break in the correlation
structure.

Batch 49: which variables, and when
-----------------------------------

The raw data are ambiguous about batch 49. ``Flow-1`` looks suspicious in the
overlay, but it is a noisy tag in every batch. The SPE contributions settle
the question. The signed contributions are the residuals of every
(tag, time) cell after the two-component reconstruction; their squares add up
to the SPE, so the squares are each cell's share of it. Summing the shares
per tag ranks the variables, and summing them per time sample locates the
event.

.. literalinclude:: dupont_batch_pca.py
   :language: python
   :start-after: # -- section: spe-49 --
   :end-before: # -- end: spe-49 --

.. code-block:: text

   Batch 49: SPE share per tag = Flow-1 3%, Flow-2 18%, Press-1 5%, Press-2 15%, Press-3 12%,
             TempC-1 19%, TempH-1 14%, TempR-1 7%, TempR-2 3%, TempR-3 4%
   Batch 49: the seven largest per-sample shares sit at samples [57, 58, 59, 60, 61, 62, 63]

``Flow-1`` carries 3% of the residual. The residual belongs to the cooling and
heating medium temperatures and to the pressures, and it is concentrated in a
short window around samples 57 to 63: a small disturbance in the heating,
cooling and pressure systems during that stretch of the batch. Nomikos and
MacGregor report that the quality of batch 49 was barely acceptable, which is
consistent with a short event rather than a batch that was wrong throughout.
:func:`process_improve.batch.unfolded_contribution_plot` draws the full
vector of 1000 bars grouped by tag, and the same data summed per tag.

The score outliers
------------------

Batches 50 to 55 are far out along the components, so the tool here is the
score contribution: how much every (tag, time) cell contributes to
:math:`t_1` or :math:`t_2`. The loading :math:`p_1`, drawn as a function of
time with :func:`process_improve.batch.time_varying_loading_plot`, shows
which parts of the batch each component describes.

.. literalinclude:: dupont_batch_pca.py
   :language: python
   :start-after: # -- section: score-outliers --
   :end-before: # -- end: score-outliers --

.. code-block:: text

   Batch 54: t1 contributions per tag = Flow-1 +5.8, Flow-2 +4.6, Press-1 +5.9, Press-2 +7.4, Press-3 +8.1,
             TempC-1 +7.4, TempH-1 +5.2, TempR-1 +7.5, TempR-2 +8.7, TempR-3 +6.9
   Batch 55: t2 contributions per tag = Flow-1 +1.4, Flow-2 +1.7, Press-1 +3.2, Press-2 +6.8, Press-3 +8.5,
             TempC-1 +6.3, TempH-1 +3.8, TempR-1 +0.8, TempR-2 +2.6, TempR-3 +0.6

Batch 54 has a high :math:`t_1` because every tag contributes in the same
direction: the whole batch ran away from the average trajectory. Batch 55
stands out on :math:`t_2` through the pressures and the cooling-medium
temperature.

Model B: exclude batches 49 to 55 and rebuild
---------------------------------------------

A reference model must describe normal operation, so the batches identified
so far are removed and the model refitted on the remaining 48 batches, now
with three components.

.. literalinclude:: dupont_batch_pca.py
   :language: python
   :start-after: # -- section: model-b --
   :end-before: # -- end: model-b --

.. code-block:: text

   Model B (48 batches): R2 per component = 0.333, 0.133, 0.085
   Model B: largest |t2| = [37, 48, 44, 46, 16, 29]; largest |t3| = [45, 39, 46, 47, 43, 14]
   Batch 39: t3 contributions per tag = Flow-1 +0.5, Flow-2 +3.5, Press-1 +0.8, Press-2 +2.3, Press-3 +6.5,
             TempC-1 +4.9, TempH-1 +1.7, TempR-1 -0.1, TempR-2 +0.9, TempR-3 +0.2

With the extreme batches gone, a second group separates on the
:math:`t_2` and :math:`t_3` plane: batches 37, 39 and 43 to 48. Batch 39 is
a representative member; its :math:`t_3` contributions point at ``Press-3``,
``TempC-1`` and ``Flow-2``, and the raw overlay shows that these batches
were run on a slightly different pressure profile. Their quality was
acceptable. They are not bad batches, they were operated differently, and a
model of normal operation should either contain enough of them to describe
that mode or leave them out; the course notes leave them out.

Model C: the reference model
----------------------------

.. literalinclude:: dupont_batch_pca.py
   :language: python
   :start-after: # -- section: model-c --
   :end-before: # -- end: model-c --

.. code-block:: text

   Model C (40 batches): R2 per component = 0.375, 0.114, 0.064
   Model C: poor-quality batches against the limits
               T2  T2 limit    SPE  SPE limit
   batch_id
   38        3.86      9.27  19.52      24.17
   40        0.80      9.27  19.95      24.17
   41        0.28      9.27  18.01      24.17
   42        0.09      9.27  23.59      24.17

The score distribution of the third model is even. Batches 38, 40, 41 and 42
are known to have had poor quality, yet all four sit inside both the
Hotelling's :math:`T^2` limit and the SPE limit: nothing in the ten
trajectories distinguishes them from the good batches. This is the lesson the
case study is built around. A model can only detect what the measurements
contain; if the cause of poor quality leaves no trace in the recorded
variables, no amount of modelling will find it, and the fix is to measure
something else.

Running the script
------------------

.. literalinclude:: dupont_batch_pca.py
   :language: python
   :start-after: def main(
   :end-before: if __name__ == "__main__":

References
----------

* Paul Nomikos and John F. MacGregor, "Multivariate SPC charts for monitoring
  batch processes", *Technometrics*, **37**, 41-59, 1995.
* Paul Nomikos and John F. MacGregor, "Monitoring batch processes using
  multiway principal component analysis", *AIChE Journal*, **40**, 1361-1375,
  1994.
* Kevin Dunn, *Latent Variable Methods* course notes (ConnectMV, 2011-2012),
  the DuPont nylon example, CC BY-SA 3.0.
