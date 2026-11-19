- **A split-plot design keeps to limits on the settings.** `generate_design(..., hard_to_change=[...],
  constraints=[...])` for the D-, A- and I-optimal designs now builds a split-plot design whose every run meets the
  constraints, a limit that names a categorical level among them, where it used to ignore `hard_to_change` with a
  warning and build an ordinary design. The built-in search starts from runs inside the limits and takes no exchange
  that leaves them, so its runs lie on its grid of levels inside the limits. `metadata["constraints_enforced"]` is
  True for such a design. The browser app (factori.al) no longer refuses a factor that is hard to change beside limits.
