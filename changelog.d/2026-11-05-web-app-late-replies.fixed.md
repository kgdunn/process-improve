- **On factori.al, a balance of the responses asked for before a rename no longer shows the old factor
  names.** A rename now takes the balance away, with any balance still on its way, as it takes the what-if
  panel away, until the analysis under the new names finds it again. (#1096)
- **On factori.al, a refusal to add runs stays on the status line.** When the reader asked for runs while
  the analysis's charts were being drawn, the charts then said "Analysis complete." over the refusal. The
  charts now leave the status line alone once something else has been said on it. (#1096)
- **The workbook factori.al downloads no longer holds the time it was written.** Its created and modified
  dates, and the date of every member of its zip archive, are now 1 January 1980, as in the `write_run_sheet`
  workbook, so two downloads of the same study are the same file; the cells are as before. (#1096)
