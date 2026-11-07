- **`BatchPCA.transform`, `diagnose` and `fit_transform` give the rows in the order of the batches
  passed**, as `BatchPLS.transform` and `predict` do. They were sorted by batch id, so a batch keyed
  `"g10"` came before `"g2"`, and a caller who lined the result up with their own list of batches
  got the wrong rows. The fitted attributes, such as `scores_`, keep their order. (#1184)
- **`batch_dtw` no longer prints a line per iteration to stdout.** The change in the weights at each
  iteration is logged at DEBUG level on the `process_improve.batch.preprocessing` logger;
  `docs/development/logging.rst` shows how to see it. `show_progress` still shows the progress
  bars, on stderr. (#1184)
