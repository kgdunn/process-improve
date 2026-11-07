- **Work asked for in one study no longer lands in the study that took its place in factori.al.**
  Runs added, results simulated, a run sheet made, or a design picked for the brief while a new
  study was started, or another opened from Your studies, went into that study when Python
  answered. A study opened meanwhile took the other study's runs and was saved with them; a new
  study received the run sheet made for the brief before it, or that brief's design; and a new
  study started while runs were added left "Adding the runs..." busy on the status line. Python's
  answer is now dropped when the study that asked for it is no longer on the page, and its busy
  note goes with it. (#1169)
- **The worked example, shown after a study whose runs give curves, no longer stops with an
  error.** The fit of the other study's curves stayed on the page, and the example's what-if panel
  looked for a curve the example does not have. A new run sheet now starts without a fit. (#1169)
