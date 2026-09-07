Batch data analysis
===================

Three case studies on batch process data. In each one the trajectories of
every batch are unfolded batchwise, one row per batch, so that PCA and PLS can
describe how the batches differ from each other, flag the abnormal ones, and
trace an event back to the variables involved and the time at which it
happened.

Each case study is a plain Python script that sits next to its page. The page
quotes the script section by section and states what each section prints; run
the script to reproduce the numbers and to get every figure as an HTML file.
The data are downloaded from `openmv.net <https://openmv.net>`_ when the
script runs.

.. toctree::
   :maxdepth: 1

   dupont-batch-pca
   sbr-batch-pls
   fmc-multiblock-batch-pls
