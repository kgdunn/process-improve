- **Browser app (factori.al): a design exported from JMP, Minitab, Design-Expert or MODDE is recognized.**
  "Analyze data you have" knows each program's .csv or .xlsx export by its headings, names the program, and leaves
  out the columns the program numbers and codes its runs with (Minitab's `CenterPt` was guessed a factor before).
  `api_read_table` returns the new keys `exported_from` and `set_aside`.
  A Design-Expert export, as text or as a workbook, opens with its lettered factors ("a:Temp") as factors, its
  Group column as the whole plot, its units, and its responses still to measure when the design was not yet run.
