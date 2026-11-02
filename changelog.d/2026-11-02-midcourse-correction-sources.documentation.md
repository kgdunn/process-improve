- **Mid-course correction: the literature attributions and the explanations are corrected against
  the sources and re-measured.** The docstrings of `batch.control`, the simulator, the
  `correct_batch_midcourse` and `evaluate_batch_control_policy` tools, the `midcourse_correction`
  recipe and the batch-control user guide now say what Flores-Cerrillo and MacGregor (2004), Yabuki
  and MacGregor (1997), Garcia-Munoz, Kourti and MacGregor (2004) and Golshan et al. (2010) did,
  and no longer say that published mid-course gains are model predictions. The gap to the
  oracle-from-k row is described as model error, foresight (the oracle runs with each batch's own
  seed) and the corrector's own limits, not model error alone; `optimal_trajectory` is described
  as a multi-start local search, a lower bound on the true optimum; and the hard-cap iteration is
  no longer called exact, since with both caps binding it can leave a statistic well inside its
  cap. Late decision points lose their gain because the model stops seeing the leverage, not
  because the process loses it: the oracle still gains 1.6 g/L on the same batches from day 7,
  and the user guide's sweep now shows the predicted, executed and oracle gains side by side. The
  model's understated gains are no longer attributed to regression toward the mean.
