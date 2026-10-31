- **On factori.al, renaming a factor while the what-if panel is being fitted no longer throws an error.** The
  fits asked for under the old name could come back after the rename was taken. The panel then looked up the
  renamed factor's setting, found none, and stopped half drawn with `Cannot read properties of undefined`. A
  rename now takes the what-if panel away, with its old fits and any still on their way, until the analysis
  under the new names fits it again. (#1069)
